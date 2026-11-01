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

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace M::MO {

#define GEN_PASS_DEF_FLATTENELEMENTWISE
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

template <typename ElemwiseInterfaceT>
bool isElemwiseOpSimple(ElemwiseInterfaceT elemwise) {
  // These ops' result types can't be changed trivially.
  if (isa<MO::RangeOp, MO::UnsqueezeShapeOp>(elemwise))
    return false;

  return true;
}

bool isSimpleOp(Operation *op) {
  if (auto unary = dyn_cast<ElementWiseUnary>(op))
    return isElemwiseOpSimple(unary);
  else if (auto binary = dyn_cast<ElementWiseBinary>(op))
    return isElemwiseOpSimple(binary);
  else if (isa<MO::Static::ReshapeOp>(op))
    return true;
  return false;
}

template <typename ElemwiseInterfaceT>
void flattenElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder) {
  TypedValue<TensorType> res = elemwise.getResult();
  TensorType resTt = res.getType();

  if (resTt.getRank() <= 1)
    return;

  MOSH::ShapeAttr shape = resTt.getShape();
  TypedAttr flattenedSize = shape.getSymbolicNumElements();
  MOSH::ShapeAttr newShape =
      MOSH::ShapeAttr::get({flattenedSize}, shape.getType());

  for (auto [i, operand] : llvm::enumerate(elemwise->getOperands())) {
    // Differs from resTt for cast / pow; pow operand dtypes may also differ.
    TensorType inpTt = cast<TensorType>(operand.getType());
    TensorType newInpTt = TensorType::get(
        newShape, inpTt.getDtype(), inpTt.getDeviceRef(), inpTt.getMetadata());

    builder.setInsertionPoint(elemwise);
    auto newReshapeOp = MO::Static::ReshapeOp::create(
        builder, elemwise.getLoc(), newInpTt, operand);

    elemwise->setOperand(i, newReshapeOp.getResult());
  }

  builder.setInsertionPointAfter(elemwise);
  auto newReshapeOp =
      MO::Static::ReshapeOp::create(builder, elemwise.getLoc(), resTt, res);
  res.replaceAllUsesExcept(newReshapeOp.getResult(), {newReshapeOp});

  TensorType newResTt = TensorType::get(
      newShape, resTt.getDtype(), resTt.getDeviceRef(), resTt.getMetadata());
  res.setType(newResTt);
}

/// Collects same-block operand producers and users of `u`.
void getNeighbors(SmallVector<Operation *> &neighbors, Operation *u) {
  neighbors.clear();
  for (auto operands : u->getOperands())
    if (auto v = operands.getDefiningOp())
      if (v->getBlock() == u->getBlock())
        neighbors.push_back(v);
  for (auto v : u->getUsers())
    if (v->getBlock() == u->getBlock())
      neighbors.push_back(v);
}

/// Collects all simple ops reachable from `simpleOp` (DFS).
llvm::SmallDenseSet<Operation *> collectSubgraph(Operation *simpleOp,
                                                 Block *block) {

  llvm::SmallDenseSet<Operation *> subgraph;
  SmallVector<Operation *> stack{simpleOp};
  llvm::SmallDenseSet<Operation *> previouslyAddedToStack{simpleOp};
  SmallVector<Operation *> neighbors;

  while (!stack.empty()) {
    Operation *u = stack.pop_back_val();

    if (!subgraph.contains(u) && isSimpleOp(u)) {
      subgraph.insert(u);

      getNeighbors(neighbors, u);
      for (auto v : neighbors) {
        if (!previouslyAddedToStack.contains(v)) {
          previouslyAddedToStack.insert(v);
          stack.push_back(v);
        }
      }
    }
  }

  return subgraph;
}

template <typename ElemwiseInterfaceT>
void handleElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder,
                      llvm::SmallDenseSet<Operation *> &visitedOps) {
  if (visitedOps.contains(elemwise))
    return;

  llvm::SmallDenseSet<Operation *> subgraph =
      collectSubgraph(elemwise, elemwise->getBlock());

  auto numReshapes = 0;
  for (auto op : subgraph)
    if (isa<MO::Static::ReshapeOp>(op))
      ++numReshapes;

  // Without reshapes, flattening unblocks no fusions.
  if (numReshapes == 0)
    return;

  for (auto op : subgraph)
    if (auto unary = dyn_cast<ElementWiseUnary>(op))
      flattenElemwiseOp(unary, builder);
    else if (auto binary = dyn_cast<ElementWiseBinary>(op))
      flattenElemwiseOp(binary, builder);

  for (auto op : subgraph)
    visitedOps.insert(op);
}

/// Flattens connected subgraphs of elementwise + reshape ops to rank 1, since
/// reshapes between elementwise ops block fusion and a common shape removes
/// the internal ones.
///
/// Reshapes inserted at the boundary with non-elementwise ops usually still
/// fuse via view decomposition + affine view fusion. Extraneous reshapes are
/// left for canonicalization + CSE.
template <typename ElemwiseInterfaceT>
void maybeFlattenElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder,
                            llvm::SmallDenseSet<Operation *> &visitedOps) {}
struct FlattenElementwise
    : public impl::FlattenElementwiseBase<FlattenElementwise> {
  using FlattenElementwiseBase::FlattenElementwiseBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module->getContext();

    OpBuilder builder{ctx};
    llvm::SmallDenseSet<Operation *> visitedOps;
    module->walk([&](ElementWiseUnary elemwise) {
      handleElemwiseOp(elemwise, builder, visitedOps);
    });
    module->walk([&](ElementWiseBinary elemwise) {
      handleElemwiseOp(elemwise, builder, visitedOps);
    });
  }
};

} // namespace M::MO
