//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"

using namespace M;
using CloneOptions = mlir::Operation::CloneOptions;

static Operation *
cloneOperation(Operation *original, BlockAndValueMapping &mapper,
               DenseMap<Operation *, Operation *> &operationMap,
               CloneOptions options);

/// Clone this region into 'dest' before the given position in 'dest'.
static void cloneRegion(Region &original, Region &dest,
                        BlockAndValueMapping &mapper,
                        DenseMap<Operation *, Operation *> &operationMap) {
  assert(&original != &dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (original.empty())
    return;

  Region::iterator destPos = dest.end();

  // The below clone implementation takes special care to be read only for the
  // sake of multi threading. That essentially means not adding any uses to any
  // of the blocks or operation results contained within this region as that
  // would lead to a write in their use-def list. This is unavoidable for
  // 'Value's from outside the region however, in which case it is not read
  // only. Using the BlockAndValueMapper it is possible to remap such 'Value's
  // to ones owned by the calling thread however, making it read only once
  // again.

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  for (Block &block : original) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest.getBlocks().insert(destPos, newBlock);
  }

  auto newBlocksRange = llvm::make_range(
      Region::iterator(mapper.lookup(&original.front())), destPos);

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them.
  // Cloning the operands and region as well would lead to uses of operations
  // not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto zippedBlocks : llvm::zip(original, newBlocksRange)) {
    Block &sourceBlock = std::get<0>(zippedBlocks);
    Block &clonedBlock = std::get<1>(zippedBlocks);
    // Clone and remap the operations within this block.
    for (Operation &op : sourceBlock)
      clonedBlock.push_back(
          cloneOperation(&op, mapper, operationMap, cloneOptions));
  }

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto zippedBlocks : llvm::zip(original, newBlocksRange)) {
    for (auto ops :
         llvm::zip(std::get<0>(zippedBlocks), std::get<1>(zippedBlocks))) {
      Operation &source = std::get<0>(ops);
      Operation &clone = std::get<1>(ops);

      operands.resize(source.getNumOperands());
      llvm::transform(
          source.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      clone.setOperands(operands);

      for (auto [sourceRegion, destRegion] :
           llvm::zip(source.getRegions(), clone.getRegions()))
        cloneRegion(sourceRegion, destRegion, mapper, operationMap);
    }
  }
}

/// Create a deep copy of this operation, remapping any operands that use
/// values outside of the operation using the map that is provided (leaving
/// them alone if no entry is present).  Replaces references to cloned
/// sub-operations to the corresponding operation that is copied, and adds
/// those mappings to the map.
static Operation *
cloneOperation(Operation *original, BlockAndValueMapping &mapper,
               DenseMap<Operation *, Operation *> &operationMap,
               CloneOptions options) {
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;

  // Remap the operands.
  if (options.shouldCloneOperands()) {
    operands.reserve(original->getNumOperands());
    for (auto opValue : original->getOperands())
      operands.push_back(mapper.lookupOrDefault(opValue));
  }

  // Remap the successors.
  successors.reserve(original->getNumSuccessors());
  for (Block *successor : original->getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto *newOp = Operation::create(
      original->getLoc(), original->getName(), original->getResultTypes(),
      operands, original->getAttrs(), successors, original->getNumRegions());
  operationMap[original] = newOp;

  // Clone the regions.
  if (options.shouldCloneRegions()) {
    for (unsigned i = 0, e = original->getNumRegions(); i != e; ++i)
      cloneRegion(original->getRegion(i), newOp->getRegion(i), mapper,
                  operationMap);
  }

  // Remember the mapping of any results.
  for (unsigned i = 0, e = original->getNumResults(); i != e; ++i)
    mapper.map(original->getResult(i), newOp->getResult(i));

  return newOp;
}

/// This is like `Operation::clone`, but instead of just keeping track of the
/// block and value mapping for the copy, it also keeps track of the
/// operation<->operation mapping.  This matters because not all operations have
/// results.
Operation *M::cloneOperation(Operation *original, BlockAndValueMapping &mapper,
                             DenseMap<Operation *, Operation *> &operationMap) {
  return ::cloneOperation(original, mapper, operationMap, CloneOptions::all());
}
