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

#include "GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace M;
using namespace MGP;

class DependencyTreeTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  OwningOpRef<mlir::ModuleOp> moduleOp;

  void SetUp() override { context.allowUnregisteredDialects(); }

  SmallVector<Operation *> parseIR(const std::string &irString) {
    moduleOp = mlir::parseSourceString<mlir::ModuleOp>(irString, &context);
    SmallVector<Operation *> ops;
    for (auto &op : moduleOp->getBody()->getOperations()) {
      ops.push_back(&op);
    }
    return ops;
  }
};

TEST_F(DependencyTreeTest, BuildTree) {
  // Ops are purposefully out of order to ensure DependencyTree doesn't rely on
  // IR ordering
  std::string irString = R"mlir(
module {
    %op2 = "op"(%op0, %op1) : (index, index) -> index
    %op0 = "op"() : () -> index
    %op1 = "op"() : () -> index
    %op3 = "op"(%op0, %op2) : (index, index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto tree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  auto op3Node = tree.get(ops[3]);
  ASSERT_EQ(op3Node->parent, nullptr);
  ASSERT_EQ(op3Node->children.size(), 1);

  auto op2Node = tree.get(ops[0]);
  ASSERT_EQ(op2Node->parent, op3Node);
  ASSERT_EQ(op2Node->children.size(), 2);

  auto op0Node = tree.get(ops[1]);
  ASSERT_EQ(op0Node->parent, op2Node);
  ASSERT_TRUE(op0Node->children.empty());

  auto op1Node = tree.get(ops[2]);
  ASSERT_EQ(op1Node->parent, op2Node);
  ASSERT_TRUE(op1Node->children.empty());

  // Test the executesAfter method with Operation* parameters
  // ops[0] is %op2, ops[1] is %op0, ops[2] is %op1, ops[3] is %op3

  // %op2 executes after %op0 and %op1
  ASSERT_TRUE(tree.executesAfter(ops[0], ops[1]));
  ASSERT_TRUE(tree.executesAfter(ops[0], ops[2]));

  // %op3 executes after all other ops
  ASSERT_TRUE(tree.executesAfter(ops[3], ops[0]));
  ASSERT_TRUE(tree.executesAfter(ops[3], ops[1]));
  ASSERT_TRUE(tree.executesAfter(ops[3], ops[2]));
}

TEST_F(DependencyTreeTest, CopyConstructorAndAssignment) {
  std::string irString = R"mlir(
module {
    %op0 = "op"() : () -> index
    %op1 = "op"(%op0) : (index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto tree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  auto verifyTree = [&](const DependencyTree &newTree) {
    ASSERT_EQ(tree.getBlock(), newTree.getBlock());
    ASSERT_NE(tree.get(ops[0]), newTree.get(ops[0]));
    ASSERT_EQ(tree.get(ops[0])->children.size(),
              newTree.get(ops[0])->children.size());
  };

  DependencyTree copiedTree(tree);
  verifyTree(copiedTree);

  DependencyTree assignedTree;
  assignedTree = tree;
  verifyTree(assignedTree);
}

TEST_F(DependencyTreeTest, MoveConstructorAndAssignment) {
  std::string irString = R"mlir(
module {
    %op0 = "op"() : () -> index
    %op1 = "op"(%op0) : (index) -> index
    %op2 = "op"(%op1) : (index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto originalTree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  auto originalOp0Node = originalTree.get(ops[0]);
  auto originalOp1Node = originalTree.get(ops[1]);
  auto originalOp2Node = originalTree.get(ops[2]);

  auto verifyTree = [&](const DependencyTree &newTree) {
    ASSERT_EQ(newTree.getBlock(), ops[0]->getBlock());
    ASSERT_EQ(newTree.get(ops[0]), originalOp0Node);
    ASSERT_EQ(newTree.get(ops[1]), originalOp1Node);
    ASSERT_EQ(newTree.get(ops[2]), originalOp2Node);
  };

  DependencyTree movedTree(std::move(originalTree));
  verifyTree(movedTree);

  DependencyTree moveAssignedTree;
  moveAssignedTree = std::move(movedTree);
  verifyTree(moveAssignedTree);
}

TEST_F(DependencyTreeTest, ExecutesAfter) {
  std::string irString = R"mlir(
module {
    %op0 = "op"() : () -> index
    %op1 = "op"(%op0) : (index) -> index
    %op2 = "op"(%op1) : (index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto tree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  ASSERT_TRUE(tree.executesAfter(ops[1], ops[0])); // %op1 executes after %op0
  ASSERT_TRUE(tree.executesAfter(ops[2], ops[1])); // %op2 executes after %op1
  ASSERT_TRUE(tree.executesAfter(ops[2], ops[0])); // %op2 executes after %op0
  ASSERT_FALSE(
      tree.executesAfter(ops[0], ops[1])); // %op0 does not execute after %op1
  ASSERT_FALSE(
      tree.executesAfter(ops[1], ops[2])); // %op1 does not execute after %op2
}

TEST_F(DependencyTreeTest, FindLCA) {
  std::string irString = R"mlir(
module {
    %op0 = "op"() : () -> index
    %op1 = "op"() : () -> index
    %op2 = "op"(%op0, %op1) : (index, index) -> index
    %op3 = "op"(%op2) : (index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto tree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  auto op0 = tree.get(ops[0]);
  auto op1 = tree.get(ops[1]);
  auto op2 = tree.get(ops[2]);
  auto op3 = tree.get(ops[3]);

  ASSERT_EQ(tree.findLCA(op0, op1), op2);
  ASSERT_EQ(tree.findLCA(op0, op2), op2);
  ASSERT_EQ(tree.findLCA(op0, op3), op3);
  ASSERT_EQ(tree.findLCA({op0, op1, op2}), op2);
}

TEST_F(DependencyTreeTest, ContainsAndGet) {
  std::string irString = R"mlir(
module {
    %op0 = "op"() : () -> index
    %op1 = "op"(%op0) : (index) -> index
}
)mlir";

  auto ops = parseIR(irString);
  auto tree = DependencyTree::getDependencyTree(ops[0]->getBlock());

  ASSERT_TRUE(tree.contains(ops[0]));
  ASSERT_TRUE(tree.contains(ops[1]));
  ASSERT_FALSE(tree.contains(nullptr));

  ASSERT_NE(tree.get(ops[0]), nullptr);
  ASSERT_NE(tree.get(ops[1]), nullptr);
  ASSERT_EQ(tree.get(nullptr), nullptr);
}
