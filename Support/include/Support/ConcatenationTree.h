//===- ConcatenationTree.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This data structure is used by clients that want to build (potentially large)
// structures by concatenating and rearranging blobs of data.  This is
// implemented by building a tree of nodes that can be concatenated together
// multiple times without copying the data around.
//
// When the final structure is finished, you can emit it to a flat form once.
//
//===----------------------------------------------------------------------===//

#ifndef CONCATENATIONTREE_H
#define CONCATENATIONTREE_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace M {
class ConcatTreeBaseNode;
/// This is the main typedef used in the codebase when referring to a tree of
/// data.
class ConcatenationTree {
public:
  ConcatenationTree();
  ConcatenationTree(ConcatenationTree &&rhs);
  ~ConcatenationTree();

  ConcatenationTree &operator=(ConcatenationTree &&rhs);

  /// Get an empty ConcatenationTree.
  static ConcatenationTree getEmpty() { return ConcatenationTree(); }

  /// Get a ContatenationTree with the specified vector data.
  static ConcatenationTree takeVector(std::vector<uint8_t> data);

  /// Concatenate and return two trees of data.
  static ConcatenationTree concat(ConcatenationTree lhs, ConcatenationTree rhs);

  /// This returns the size in bytes of the collection of data that this
  /// represents.  This is O(1).
  size_t getSize() const;

  /// Iterate through the ConcatenationTree walking over the leaf nodes with
  /// data in-order.  The specified TraversalFn is expected to be a callable
  /// that takes `ArrayRef<uint8_t>` and returns void.
  void traverse(std::function<void(ArrayRef<uint8_t>)> fn);

private:
  ConcatenationTree(ConcatTreeBaseNode *nodePtr);
  std::unique_ptr<ConcatTreeBaseNode> node;
};

} // namespace M

#endif // CONCATENATIONTREE_H