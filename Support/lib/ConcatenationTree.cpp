//===- ConcatenationTree.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ConcatenationTree.h"
using namespace M;

//===----------------------------------------------------------------------===//
// ConcatTreeBaseNode
//===----------------------------------------------------------------------===//

/// This data structure provides the ability to concatenate (potentially large)
/// strings together without copying the data around too much.  This is the
/// abstract base class.
namespace M {
class ConcatTreeBaseNode {
public:
  const enum NodeKind {
    kVector, // a node holding an std::vector of data.
    kBranch, // a node with two or more subnodes.
  } nodeKind;

  virtual ~ConcatTreeBaseNode();

protected:
  ConcatTreeBaseNode(NodeKind kind) : nodeKind(kind) {}
};
} // namespace M

ConcatTreeBaseNode::~ConcatTreeBaseNode() {}

//===----------------------------------------------------------------------===//
// ConcatTreeVectorNode
//===----------------------------------------------------------------------===//

namespace {
class ConcatTreeVectorNode : public ConcatTreeBaseNode {
public:
  static bool classof(const ConcatTreeBaseNode *base) {
    return base->nodeKind == kVector;
  }

  ConcatTreeVectorNode(std::vector<uint8_t> data)
      : ConcatTreeBaseNode(kVector), data(std::move(data)) {}

  std::vector<uint8_t> data;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConcatTreeBranchNode
//===----------------------------------------------------------------------===//

namespace {
class ConcatTreeBranchNode : public ConcatTreeBaseNode {
public:
  ConcatTreeBranchNode(ConcatenationTree lhs, ConcatenationTree rhs)
      : ConcatTreeBaseNode(kBranch) {
    child[0] = std::move(lhs);
    child[1] = std::move(rhs);
    totalSize = child[0].getSize() + child[1].getSize();
  }

  static bool classof(const ConcatTreeBaseNode *base) {
    return base->nodeKind == kBranch;
  }

  // We support up to four childen in this, reducing the number of tree nodes
  // that get created.
  ConcatenationTree child[4];
  size_t totalSize;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConcatenationTree implementation logic
//===----------------------------------------------------------------------===//

ConcatenationTree::ConcatenationTree() {}
ConcatenationTree::ConcatenationTree(ConcatTreeBaseNode *nodePtr)
    : node(nodePtr) {}
ConcatenationTree::ConcatenationTree(ConcatenationTree &&rhs) = default;
ConcatenationTree &
ConcatenationTree::operator=(ConcatenationTree &&rhs) = default;

ConcatenationTree::~ConcatenationTree() {}

/// Get a ContatenationTree with the specified vector data.
ConcatenationTree ConcatenationTree::takeVector(std::vector<uint8_t> data) {
  if (data.empty())
    return getEmpty();

  return new ConcatTreeVectorNode(std::move(data));
}

/// Concatenate and return two trees of data.
ConcatenationTree ConcatenationTree::concat(ConcatenationTree lhs,
                                            ConcatenationTree rhs) {
  // Collapse null trees away.
  if (!lhs.node)
    return rhs;
  if (!rhs.node)
    return lhs;

  // Concat nodes have extra space in them that we can fill up to avoid
  // allocating new concat nodes.

  // If the left side is a branch node with space, we can add nodes to it
  // instead of allocating another branch.
  if (auto *lhsConcat = dyn_cast<ConcatTreeBranchNode>(lhs.node.get())) {
    if (!lhsConcat->child[2].node) {
      lhsConcat->totalSize += rhs.getSize();
      lhsConcat->child[2] = std::move(rhs);
      return lhs;
    }
    if (!lhsConcat->child[3].node) {
      lhsConcat->totalSize += rhs.getSize();
      lhsConcat->child[3] = std::move(rhs);
      return lhs;
    }
  }

  // If the right side is a concat node with space, we can push into it.
  if (auto *rhsConcat = dyn_cast<ConcatTreeBranchNode>(rhs.node.get())) {
    if (!rhsConcat->child[2].node) {
      rhsConcat->totalSize += lhs.getSize();
      // Move everything down so we can insert to the left of them.
      rhsConcat->child[2] = std::move(rhsConcat->child[1]);
      rhsConcat->child[1] = std::move(rhsConcat->child[0]);
      rhsConcat->child[0] = std::move(lhs);
      return rhs;
    }
    if (!rhsConcat->child[3].node) {
      rhsConcat->totalSize += lhs.getSize();
      // Move everything down so we can insert to the left of them.
      rhsConcat->child[3] = std::move(rhsConcat->child[2]);
      rhsConcat->child[2] = std::move(rhsConcat->child[1]);
      rhsConcat->child[1] = std::move(rhsConcat->child[0]);
      rhsConcat->child[0] = std::move(lhs);
      return rhs;
    }
  }

  return new ConcatTreeBranchNode(std::move(lhs), std::move(rhs));
}

/// This returns the size in bytes of the collection of data that this
/// represents.  This is O(1).
size_t ConcatenationTree::getSize() const {
  ConcatTreeBaseNode *nodePtr = node.get();
  if (nodePtr == nullptr)
    return 0;

  if (auto *vec = dyn_cast<ConcatTreeVectorNode>(nodePtr))
    return vec->data.size();

  return cast<ConcatTreeBranchNode>(nodePtr)->totalSize;
}

void ConcatenationTree::traverse(std::function<void(ArrayRef<uint8_t>)> fn) {
  ConcatTreeBaseNode *nodePtr = node.get();
  if (nodePtr == nullptr)
    return;

  if (auto *vec = dyn_cast<ConcatTreeVectorNode>(nodePtr))
    return fn(vec->data);

  auto *branch = cast<ConcatTreeBranchNode>(nodePtr);
  branch->child[0].traverse(fn);
  branch->child[1].traverse(fn);

  // Child #2/3 are optional, traverse if present.
  if (branch->child[2].node) {
    branch->child[2].traverse(fn);
    if (branch->child[3].node)
      branch->child[3].traverse(fn);
  }
}
