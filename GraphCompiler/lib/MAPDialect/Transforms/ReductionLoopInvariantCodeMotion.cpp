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
//
// Moves the work of a rowwise kernel's map phases that is the same for the
// whole row to row level, so it runs once per row instead of once per element:
//
//   map.iter.map index (%i) {           %inv = rsqrt(%acc)
//     %inv = rsqrt(%acc)          =>    map.iter.map index (%i) {
//     %y = mul(%x[%i], %inv)              %y = mul(%x[%i], %inv)
//   }                                   }
//
// That includes the work in a phase's regions, such as a concat select's,
// which read the same row.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_REDUCTIONLOOPINVARIANTCODEMOTION
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Whether `value` is computed outside `phase`, once per row.
bool isRowValue(IterMapOp phase, Value value) {
  return !phase.getBody().isAncestor(value.getParentRegion());
}

// Re-indexes `load` by the row, before `phase`, when its index ignores the
// reduced axis. The index may move along that axis first, as in a concat
// select's region.
void hoistLoad(IterRowwiseOp dst, IterMapOp phase, LoadOp load,
               RewriterBase &rewriter) {
  std::optional<SmallVector<int64_t>> dims =
      IndexTransformOpInterface::traceLoopDims(
          load.getIndex(), phase.getIndexArg(), dst.getAxis());
  if (!dims || llvm::is_contained(*dims, dst.getAxis()) ||
      !isRowValue(phase, load.getTensor()))
    return;
  SmallVector<Operation *> chain;
  Value root = load.getIndex();
  for (Operation *def = root.getDefiningOp(); def; def = root.getDefiningOp()) {
    chain.push_back(def);
    root = cast<IndexTransformOpInterface>(def).getIndex();
  }
  // The phase's index or a select region's argument: any index in the row.
  IRMapping toRow;
  toRow.map(root, dst.getRowArg());
  for (Operation *def : llvm::reverse(chain))
    rewriter.clone(*def, toRow);
  rewriter.replaceOp(load, rewriter.clone(*load, toRow)->getResults());
}

// Moves `call` before `phase` when every operand is a row value.
void hoistCall(IterRowwiseOp dst, IterMapOp phase, CallElementwiseOp call) {
  if (call.readsIndex() || !llvm::all_of(call.getArgs(), [&](Value arg) {
        return isRowValue(phase, arg);
      }))
    return;
  call->moveBefore(phase);
  call.getIndexMutable().assign(dst.getRowArg());
}

// Erases the ops of `region` and its nested regions that nothing reads.
void eraseDeadOps(Region &region, RewriterBase &rewriter) {
  for (Operation &op : llvm::make_early_inc_range(
           llvm::reverse(region.front().getOperations()))) {
    for (Region &inner : op.getRegions())
      eraseDeadOps(inner, rewriter);
    if (isOpTriviallyDead(&op))
      rewriter.eraseOp(&op);
  }
}

// Hoists `phase`'s row-invariant ops in order, so an op whose operands were
// hoisted before it follows them. That includes the ops in its regions, such
// as a concat select's.
void hoistRowInvariant(IterRowwiseOp dst, IterMapOp phase,
                       RewriterBase &rewriter) {
  SmallVector<Operation *> ops;
  phase.getBody().walk<WalkOrder::PreOrder>(
      [&](Operation *op) { ops.push_back(op); });
  for (Operation *op : ops) {
    rewriter.setInsertionPoint(phase);
    if (isa<ConstantScalarOp>(op))
      op->moveBefore(phase);
    else if (auto load = dyn_cast<LoadOp>(op))
      hoistLoad(dst, phase, load, rewriter);
    else if (auto call = dyn_cast<CallElementwiseOp>(op))
      hoistCall(dst, phase, call);
  }
  eraseDeadOps(phase.getBody(), rewriter);
}

struct ReductionLoopInvariantCodeMotionPass
    : public impl::ReductionLoopInvariantCodeMotionBase<
          ReductionLoopInvariantCodeMotionPass> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation().walk([&](IterRowwiseOp dst) {
      for (IterMapOp phase : llvm::make_early_inc_range(
               dst.getBody().front().getOps<IterMapOp>()))
        hoistRowInvariant(dst, phase, rewriter);
    });
  }
};

} // namespace

} // namespace M::MAP
