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

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"

namespace M::MO {

#define GEN_PASS_DEF_INFERLAYOUTS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

//===----------------------------------------------------------------------===//
// InferLayouts pass.
//===----------------------------------------------------------------------===//

struct InferLayoutPass : public impl::InferLayoutsBase<InferLayoutPass> {
  /// Maps `value` to `layout`, emitting an error on `op` (the constraint's
  /// source) if `value` is already mapped to a different layout. A no-op when
  /// `value`'s type already has a layout, so the pass is idempotent.
  LogicalResult addLayoutConstraint(Operation *op, TensorValue value,
                                    LayoutAttr layout) {
    if (value.getType().getLayoutAttr())
      return success();

    auto [iter, inserted] = valueToInferredLayout.insert({value, layout});
    if (!inserted && iter->getSecond() != layout) {
      return op->emitError()
             << "attempting to constrain value to layout: " << layout
             << " when it's already constrained to layout: "
             << iter->getSecond();
    }

    return success();
  }

  llvm::StringRef getConvFilterLayout(size_t rank) {
    switch (rank) {
    case 4:
      return "FCRS";
    case 5:
      return "FCQRS";
    default:
      llvm_unreachable("Only able to handle convs up to conv1-3d");
    }
  }

  llvm::StringRef getConvInputLayout(size_t rank) {
    switch (rank) {
    case 4:
      return "NHWC";
    case 5:
      return "NDHWC";
    default:
      llvm_unreachable("Only able to handle convs up to conv1-3d");
    }
  }

  llvm::StringRef getConvTransposeFilterLayout(size_t rank) {
    switch (rank) {
    case 4:
      return "RSFC";
    case 5:
      return "QRSFC";
    default:
      llvm_unreachable("Only able to handle convTranspose 2-3d");
    }
  }

  bool isSupportedConv(size_t inputRank, size_t filterRank) {
    return (inputRank == 4 && filterRank == 4) ||
           (inputRank == 5 && filterRank == 5);
  }

  /// Seeds `valueToInferredLayout` from ops with direct layout constraints
  /// (e.g. `mo.conv` requires an `NHWC` input). Fails on conflicting
  /// constraints.
  LogicalResult addSeedingConstraints() {
    LayoutAttr matmulWeightsLayout = LayoutAttr::get(ctx, "KN");

    WalkResult walkResult =
        graphOp.walk([&](Operation *op) {
          if (auto convOp = dyn_cast<ConvOp>(op)) {
            TensorValue input = convOp.getInput();
            TensorValue filter = convOp.getFilter();
            auto inputRank = input.getType().getRank();
            auto filterRank = filter.getType().getRank();

            if (isSupportedConv(inputRank, filterRank)) {
              auto convInputLayout = LayoutAttr::get(
                  ctx, getConvInputLayout(input.getType().getRank()));
              auto convFilterLayout = LayoutAttr::get(
                  ctx, getConvFilterLayout(filter.getType().getRank()));

              if (failed(addLayoutConstraint(op, input, convInputLayout)) ||
                  failed(addLayoutConstraint(op, filter, convFilterLayout)))
                return WalkResult::interrupt();
            } else {
              op->emitError() << "Only support 2D and 3D layout for ConvOp.\n";
              return WalkResult::interrupt();
            }
          } else if (auto convTransposeOp = dyn_cast<ConvTransposeOp>(op)) {
            TensorValue input = convTransposeOp.getInput();
            TensorValue filter = convTransposeOp.getFilter();
            auto inputRank = input.getType().getRank();
            auto filterRank = filter.getType().getRank();

            if (isSupportedConv(inputRank, filterRank)) {
              // ConvTranspose input uses the same layouts as conv.
              auto convTransposeInputLayout = LayoutAttr::get(
                  ctx, getConvInputLayout(input.getType().getRank()));
              auto convTransposeFilterLayout = LayoutAttr::get(
                  ctx,
                  getConvTransposeFilterLayout(filter.getType().getRank()));

              if (failed(addLayoutConstraint(op, input,
                                             convTransposeInputLayout)) ||
                  failed(addLayoutConstraint(op, filter,
                                             convTransposeFilterLayout)))
                return WalkResult::interrupt();
            } else {
              op->emitError()
                  << "Only support 2D and 3D layout for ConvTransposeOp.\n";
              return WalkResult::interrupt();
            }
          } else if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
            TensorValue weights = matmulOp.getTensorB();
            if (failed(addLayoutConstraint(op, weights, matmulWeightsLayout)))
              return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });
    return failure(walkResult.wasInterrupted());
  }

  /// Propagate layout through ops (e.g., from input to output or vice-versa for
  /// elementwise ops, which requires same layouts for all inputs and outputs).
  LogicalResult propagateLayoutThroughOps() {
    SmallVector<TensorValue> worklist;
    DenseSet<TensorValue> worklistHistory;

    auto tryPushOntoWorklist = [&](TensorValue value) {
      if (auto [_, inserted] = worklistHistory.insert(value); inserted)
        worklist.push_back(value);
    };
    for (auto [value, _] : valueToInferredLayout)
      tryPushOntoWorklist(value);

    while (!worklist.empty()) {
      TensorValue value = worklist.pop_back_val();
      LayoutAttr layout = valueToInferredLayout.at(value);

      SmallVector<Operation *> affectedOps(value.getUsers());
      if (Operation *definingOp = value.getDefiningOp(); definingOp)
        affectedOps.push_back(definingOp);

      for (Operation *op : affectedOps) {
        // Could also propagate through ops like transpose.
        SmallVector<Value> valsToPropagate = [](Operation *op) {
          SmallVector<Value> results;
          if (isa<MO::GuardOp>(op)) {
            llvm::append_range(results, op->getOperands().drop_front());
            llvm::append_range(results, op->getResults());
          } else if (isa<MO::TransferOp>(op) ||
                     op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>()) {
            llvm::append_range(results, op->getOperands());
            llvm::append_range(results, op->getResults());
          }
          return results;
        }(op);

        for (Value val : valsToPropagate) {
          if (auto tensorVal = dyn_cast<TensorValue>(val)) {
            if (failed(addLayoutConstraint(op, tensorVal, layout)))
              return failure();
            tryPushOntoWorklist(tensorVal);
          }
        }
      }
    }
    return success();
  }

  void assignInferredLayoutsToTypes() {
    for (auto [value, layout] : valueToInferredLayout)
      value.setType(value.getType().cloneWith(layout));

    // Graph inputs or outputs may have new layouts.
    synchronizeMetadataUpdates(graphOp);
  }

  void runOnOperation() override {
    this->ctx = &getContext();
    this->graphOp = getOperation();

    if (failed(addSeedingConstraints()))
      return signalPassFailure();

    // This is needed to tame the MO type system.
    if (failed(propagateLayoutThroughOps()))
      return signalPassFailure();

    assignInferredLayoutsToTypes();

    valueToInferredLayout.clear();
    this->graphOp = {};
  }

  MLIRContext *ctx;
  GraphOp graphOp;
  // Only values reached by inference; most tensors are absent.
  DenseMap<TensorValue, LayoutAttr> valueToInferredLayout;
};

} // namespace M::MO
