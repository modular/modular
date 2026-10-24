//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"

#include "llvm/ADT/SparseBitVector.h"

namespace M::MGP {

struct DependencyTreeNode {
  Operation *op;
  DependencyTreeNode *parent{nullptr};
  SmallVector<DependencyTreeNode *> children;

  int preIndex{-1};
  int postIndex{-1};
  int depth{-1};

  inline void addChild(DependencyTreeNode *child) {
    children.push_back(child);
    child->parent = this;
  }

  explicit DependencyTreeNode(Operation *operation) : op(operation) {}
};

/// Dependency tree over a block's operations. A node's ancestors are the ops
/// that depend on it, i.e. are guaranteed to execute after it.
///
/// Built once by iterative dataflow analysis (O(N^2) worst case) and not
/// updated when the block changes. Dominance queries are O(1); the LCA of 2
/// nodes is O(h) and of Q nodes O(Q*h), for tree height h.
class DependencyTree {
public:
  DependencyTree();
  DependencyTree(const DependencyTree &other);
  DependencyTree &operator=(const DependencyTree &other);
  DependencyTree(DependencyTree &&other) noexcept;
  DependencyTree &operator=(DependencyTree &&other) noexcept;

  // Builds the dependency tree for the given block starting with the block
  // terminator
  static DependencyTree getDependencyTree(Block *block);

  // Dumps the tree as a GraphViz dot file, with dashed red edges
  // indicating uses of allocs
  void dumpToDot(llvm::raw_ostream &os) const;

  // Returns nullptr if the op is not in the tree.
  DependencyTreeNode *get(Operation *) const;

  Block *getBlock() const;

  bool contains(Operation *) const;

  // Returns true if a is guaranteed to execute after b
  bool executesAfter(Operation *a, Operation *b) const;

  // Returns true if all elements of a are guaranteed to execute after all
  // elements of b.
  bool executesAfter(ArrayRef<Operation *> a, ArrayRef<Operation *> b) const;

  // Returns the topological execution index of op (lower = executes earlier).
  // op must be contained in this tree.
  unsigned getTopoIndex(Operation *op) const;

  // Find the least common ancestor of two nodes in the tree.
  DependencyTreeNode *findLCA(DependencyTreeNode *a,
                              DependencyTreeNode *b) const;

  // Find the LCA of a list of nodes by repeatedly applying the above function
  DependencyTreeNode *findLCA(ArrayRef<DependencyTreeNode *> nodes) const;
  DependencyTreeNode *findLCA(ArrayRef<Operation *> nodes) const;

private:
  DependencyTreeNode *root{nullptr};
  DenseMap<Operation *, std::unique_ptr<DependencyTreeNode>> nodes;

  /// Bit vector matrix, where runsAfter[i][j] is true if operation j runs after
  /// operation i. Used for executesAfter queries.
  SmallVector<llvm::SparseBitVector<64>> runsAfter;

  /// Pre-computed counts of runsAfter sets, for computing immediate
  /// runsAfters.
  SmallVector<uint32_t> runsAfterCounts;

  // Map from operations to indices in the runsAfter sets
  DenseMap<Operation *, unsigned> operationToIndex;
  SmallVector<Operation *> indexToOperation;

  // Clones a node including its children
  DependencyTreeNode *cloneNode(DependencyTreeNode *node,
                                DependencyTreeNode *parent);

  DependencyTreeNode *getOrCreate(Operation *);
};

// Return a list of alloc uses including uses of aliases. If aliases cannot be
// tracked, return an error.
ErrorOr<SmallVector<Operation *>> getAllocUses(MGP::BufferAllocOp allocOp);

// Analysis pass to do dataflow analysis on buffer uses to assist
// with memory planning
struct BufferLifetimeAnalysisPass {
  BufferLifetimeAnalysisPass(Operation *op);

  // Association between blocks and their corresponding dependency tree.
  // This analysis hands out references, so trees are held behind a
  // unique_ptr to ensure reference stability.
  DenseMap<Block *, std::unique_ptr<DependencyTree>> dominatorTrees;

  DependencyTree &getDepTree(Block *b) {
    if (auto it = dominatorTrees.find(b); it != dominatorTrees.end())
      return *it->second;

    auto newTree =
        std::make_unique<DependencyTree>(DependencyTree::getDependencyTree(b));
    return *dominatorTrees.try_emplace(b, std::move(newTree)).first->second;
  }
};

} // namespace M::MGP
