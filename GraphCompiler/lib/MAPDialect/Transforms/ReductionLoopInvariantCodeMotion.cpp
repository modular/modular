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
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
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

// Whether `call`'s kernel reads the element's position.
bool readsIndex(CallElementwiseOp call) {
  auto decl = SymbolTable::lookupNearestSymbolFrom<MOGG::KernelDeclOp>(
      call, call.getCalleeAttr());
  return !decl ||
         MOGG::KernelDeclAdaptor(decl).elementwiseFunction->takesIndexList();
}

// Re-indexes `load` by the row, before `phase`, when its index ignores the
// reduced axis.
void hoistLoad(IterRowwiseOp dst, IterMapOp phase, LoadOp load,
               RewriterBase &rewriter) {
  std::optional<SmallVector<int64_t>> dims =
      traceLoopDims(load.getIndex(), phase.getIndexArg());
  if (!dims || llvm::is_contained(*dims, dst.getAxis()) ||
      !isRowValue(phase, load.getTensor()))
    return;
  IRMapping toRow;
  toRow.map(phase.getIndexArg(), dst.getRowArg());
  SmallVector<Operation *> chain;
  for (Operation *def = load.getIndex().getDefiningOp(); def;
       def = cast<IndexTransformOpInterface>(def).getIndex().getDefiningOp())
    chain.push_back(def);
  for (Operation *def : llvm::reverse(chain))
    rewriter.clone(*def, toRow);
  rewriter.replaceOp(load, rewriter.clone(*load, toRow)->getResults());
}

// Moves `call` before `phase` when every operand is a row value.
void hoistCall(IterRowwiseOp dst, IterMapOp phase, CallElementwiseOp call) {
  if (readsIndex(call) || !llvm::all_of(call.getArgs(), [&](Value arg) {
        return isRowValue(phase, arg);
      }))
    return;
  call->moveBefore(phase);
  call.getIndexMutable().assign(dst.getRowArg());
}

// Hoists `phase`'s row-invariant ops in order, so an op whose operands were
// hoisted before it follows them.
void hoistRowInvariant(IterRowwiseOp dst, IterMapOp phase,
                       RewriterBase &rewriter) {
  for (Operation &op : llvm::make_early_inc_range(phase.getBody().front())) {
    rewriter.setInsertionPoint(phase);
    if (isa<ConstantScalarOp>(op))
      op.moveBefore(phase);
    else if (auto load = dyn_cast<LoadOp>(op))
      hoistLoad(dst, phase, load, rewriter);
    else if (auto call = dyn_cast<CallElementwiseOp>(op))
      hoistCall(dst, phase, call);
  }
  for (Operation &op : llvm::make_early_inc_range(
           llvm::reverse(phase.getBody().front().getOperations())))
    if (isOpTriviallyDead(&op))
      rewriter.eraseOp(&op);
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
