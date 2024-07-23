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

static bool areBlocksSameImpl(Block &block1, Block &block2,
                              DenseMap<Value, llvm::hash_code> &valueHashes,
                              size_t &posInBlock) {
  llvm::hash_code block1Hash{42}, block2Hash{42};

  if (block1.getNumArguments() != block2.getNumArguments())
    return false;

  // Hash the inputs to the block.
  for (auto [idx, args] : llvm::enumerate(
           llvm::zip(block1.getArguments(), block2.getArguments()))) {
    Value &arg1 = std::get<0>(args);
    Value &arg2 = std::get<1>(args);
    // Create a unique argument based on the type and order of arguments, but
    // not the specific argument name directly.
    llvm::hash_code argHash = llvm::hash_combine(arg1.getType(), idx);
    valueHashes[arg1] = argHash;
    block1Hash = llvm::hash_combine(block1Hash, argHash);

    llvm::hash_code arg2Hash = llvm::hash_combine(arg2.getType(), idx);
    block2Hash = llvm::hash_combine(block2Hash, arg2Hash);
    valueHashes[arg2] = arg2Hash;
  }

  if (block1Hash != block2Hash)
    return false;

  // Hash all the operations in the block.
  for (const auto &[op1, op2] :
       llvm::zip(block1.getOperations(), block2.getOperations())) {
    llvm::hash_code op1Hash = mlir::OperationEquivalence::computeHash(
        &op1,
        /*hashOperands=*/[&](Value v) { return valueHashes.at(v); },
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
    op1Hash = llvm::hash_combine(op1Hash, posInBlock);

    llvm::hash_code op2Hash = mlir::OperationEquivalence::computeHash(
        &op2,
        /*hashOperands=*/[&](Value v) { return valueHashes.at(v); },
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
    op2Hash = llvm::hash_combine(op2Hash, posInBlock);

    posInBlock++;
    if (op1Hash != op2Hash)
      return false;

    // The operation hash includes the result types but we also want to get a
    // unique hash for each result. This so that users of an op with multiple
    // returns produce different hashes based on which they use.
    for (auto [idx, val] : llvm::enumerate(op1.getResults()))
      valueHashes[val] = llvm::hash_combine(op1Hash, llvm::hash_code(idx));
    for (auto [idx, val] : llvm::enumerate(op2.getResults()))
      valueHashes[val] = llvm::hash_combine(op2Hash, llvm::hash_code(idx));

    // Check all the blocks within this block are hashed the same way.
    for (const auto &[r1, r2] : llvm::zip(op1.getRegions(), op2.getRegions())) {
      for (const auto &[bb1, bb2] : llvm::zip(r1.getBlocks(), r2.getBlocks())) {
        if (!areBlocksSameImpl(bb1, bb2, valueHashes, posInBlock))
          return false;
      }
    }
  }
  return true;
}

namespace M {

llvm::hash_code hashBlock(mlir::Block &block) {
  DenseMap<Value, llvm::hash_code> valueHashes;
  size_t posInBlock = 0;
  return hashBlockImpl(block, valueHashes, posInBlock);
}

bool areBlocksSame(mlir::Block &b1, mlir::Block &b2) {
  DenseMap<Value, llvm::hash_code> valueHashes;
  size_t posInBlock = 0;
  return areBlocksSameImpl(b1, b2, valueHashes, posInBlock);
}

} // namespace M
