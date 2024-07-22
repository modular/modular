//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HashUtils.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;

static llvm::hash_code
hashBlockImpl(Block &block, DenseMap<Value, llvm::hash_code> &valueHashes,
              size_t &posInBlock) {
  llvm::hash_code blockHash{42};

  // Hash the inputs to the block.
  for (auto [idx, arg] : llvm::enumerate(block.getArguments())) {
    // Create a unique argument based on the type and order of arguments, but
    // not the specific argument name directly.
    llvm::hash_code argHash = llvm::hash_combine(arg.getType(), idx);
    valueHashes[arg] = argHash;
    blockHash = llvm::hash_combine(blockHash, argHash);
  }

  // Hash all the operations in the block.
  for (Operation &op : block.getOperations()) {
    llvm::hash_code opHash = mlir::OperationEquivalence::computeHash(
        &op,
        /*hashOperands=*/[&](Value v) { return valueHashes.at(v); },
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
    opHash = llvm::hash_combine(opHash, posInBlock++);

    // The operation hash includes the result types but we also want to get a
    // unique hash for each result. This so that users of an op with multiple
    // returns produce different hashes based on which they use.
    for (auto [idx, val] : llvm::enumerate(op.getResults()))
      valueHashes[val] = llvm::hash_combine(opHash, llvm::hash_code(idx));

    blockHash = llvm::hash_combine(blockHash, opHash);

    // Hash any blocks this op contains.
    for (Region &region : op.getRegions()) {
      for (Block &bb : region.getBlocks()) {
        blockHash = llvm::hash_combine(
            blockHash, hashBlockImpl(bb, valueHashes, posInBlock));
      }
    }
  }
  return blockHash;
}

namespace M {

llvm::hash_code hashBlock(mlir::Block &block) {
  DenseMap<Value, llvm::hash_code> valueHashes;
  size_t posInBlock = 0;
  return hashBlockImpl(block, valueHashes, posInBlock);
}

} // namespace M
