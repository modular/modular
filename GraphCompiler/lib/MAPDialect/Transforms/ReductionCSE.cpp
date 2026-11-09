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
// Common subexpression elimination across the phases of a rowwise kernel. A
// value every phase computes from the reduced input is computed once per
// element in a `map.iter.cache`, which the Row caches, and each phase reads it
// back instead of computing it again:
//
//   map.iter.reduce index (%i) {          %c = map.iter.cache index (%k) {
//     %v = mul(%x[%i], %s[%i])              %v = mul(%x[%k], %s[%k])
//     ...                           =>      map.yield %v
//   }                                     }
//   map.iter.map index (%j) {             map.iter.reduce index (%i) {
//     %v = mul(%x[%j], %s[%j])              %v = map.iter.cache.load %c[%i]
//     ...                                   ...
//   }                                     }
//                                         map.iter.map index (%j) {
//                                           %v = map.iter.cache.load %c[%j]
//                                           ...
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "Support/AssertStream.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_REDUCTIONCSE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Element expressions: values computed per element from loads at their
// phase's index (or a view of it), constants and elementwise calls.
// Memoized, since a value may be reused many times in one expression.
class ElementExprs {
public:
  // Whether `v` is an element expression at `index`.
  bool isElement(Value v, Value index) {
    if (auto it = element.find(v); it != element.end())
      return it->second;
    Operation *def = v.getDefiningOp();
    bool result = false;
    if (v == index)
      result = true;
    else if (auto view = dyn_cast_if_present<IndexTransformOpInterface>(def))
      result = def->getNumOperands() == 1 && isElement(view.getIndex(), index);
    else if (auto load = dyn_cast_if_present<LoadOp>(def))
      result = isElement(load.getIndex(), index);
    else if (isa_and_present<ConstantScalarOp>(def))
      result = true;
    else if (auto call = dyn_cast_if_present<CallElementwiseOp>(def))
      result = call.getIndex() == index &&
               llvm::all_of(call.getArgs(),
                            [&](Value arg) { return isElement(arg, index); });
    return element[v] = result;
  }

  // Whether element expressions `a`, at `aIndex`, and `b`, at `bIndex`,
  // compute the same value.
  bool same(Value a, Value aIndex, Value b, Value bIndex) {
    if (a == b)
      return true;
    if (auto it = equal.find({a, b}); it != equal.end())
      return it->second;
    Operation *defA = a.getDefiningOp(), *defB = b.getDefiningOp();
    bool result =
        defA && defB && defA->getName() == defB->getName() &&
        defA->getAttrDictionary() == defB->getAttrDictionary() &&
        a.getType() == b.getType() &&
        defA->getNumOperands() == defB->getNumOperands() &&
        llvm::all_of(llvm::zip(defA->getOperands(), defB->getOperands()),
                     [&](auto operands) {
                       auto [operandA, operandB] = operands;
                       if (operandA == aIndex && operandB == bIndex)
                         return true;
                       return same(operandA, aIndex, operandB, bIndex);
                     });
    return equal[{a, b}] = result;
  }

private:
  llvm::DenseMap<Value, bool> element;
  llvm::DenseMap<std::pair<Value, Value>, bool> equal;
};

// The ops of `phase` that compute `v`.
llvm::SetVector<Operation *> exprOps(Value v, RowwisePhaseOpInterface phase) {
  llvm::SetVector<Operation *> ops;
  SmallVector<Value> worklist = {v};
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (def && phase->isProperAncestor(def) && ops.insert(def))
      llvm::append_range(worklist, def->getOperands());
  }
  return ops;
}

bool isElementLoad(Operation *op, Value tensor, Value index) {
  auto load = dyn_cast<LoadOp>(op);
  return load && load.getTensor() == tensor && load.getIndex() == index;
}

// Whether `phase`, once it reads `copies` from the cache, no longer loads
// `tensor` at its index.
bool readsOnlyThrough(RowwisePhaseOpInterface phase, Value tensor,
                      ArrayRef<Value> copies) {
  SmallVector<Value> worklist;
  phase->walk([&](Operation *op) {
    if (op != phase && !wouldOpBeTriviallyDead(op))
      llvm::append_range(worklist, op->getOperands());
  });
  llvm::DenseSet<Operation *> live;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (llvm::is_contained(copies, v) || !def ||
        !phase->isProperAncestor(def) || !live.insert(def).second)
      continue;
    if (isElementLoad(def, tensor, phase.getIndexArg()))
      return false;
    llvm::append_range(worklist, def->getOperands());
  }
  return true;
}

struct CachedValue {
  // An element expression of the first fold.
  Value value;
  // Its copies in every phase, `value` included.
  SmallVector<Value> copies;
};

// The value the row caches. Of the first fold's element expressions over
// `tensor` that every phase computes wherever it reads `tensor`, take the one
// with the narrowest type, then the most ops: a wider one would grow the
// cached row, a smaller one would recompute work in every phase. The load
// itself always qualifies.
std::optional<CachedValue>
chooseCachedValue(ArrayRef<RowwisePhaseOpInterface> phases, Value tensor) {
  RowwisePhaseOpInterface first = phases.front();
  ElementExprs exprs;
  std::optional<CachedValue> best;
  unsigned bestBits = 0, bestSize = 0;
  for (Operation &op : first->getRegion(0).front().without_terminator()) {
    for (Value value : op.getResults()) {
      if (!value.getType().isIntOrFloat() ||
          !exprs.isElement(value, first.getIndexArg()))
        continue;
      llvm::SetVector<Operation *> ops = exprOps(value, first);
      if (llvm::none_of(ops, [&](Operation *o) {
            return isElementLoad(o, tensor, first.getIndexArg());
          }))
        continue;
      CachedValue candidate{value, {}};
      bool everywhere = llvm::all_of(phases, [&](RowwisePhaseOpInterface p) {
        size_t begin = candidate.copies.size();
        p->walk([&](Operation *o) {
          for (Value w : o->getResults())
            if (exprs.same(value, first.getIndexArg(), w, p.getIndexArg()))
              candidate.copies.push_back(w);
        });
        return readsOnlyThrough(p, tensor,
                                ArrayRef(candidate.copies).drop_front(begin));
      });
      if (!everywhere)
        continue;
      unsigned bits = value.getType().getIntOrFloatBitWidth();
      unsigned size = ops.size();
      if (!best || bits < bestBits || (bits == bestBits && size > bestSize)) {
        best = std::move(candidate);
        bestBits = bits;
        bestSize = size;
      }
    }
  }
  return best;
}

// Computes `cached`'s value in a `map.iter.cache` at the start of `row`, and
// makes each copy read it.
void createCache(RowwiseOpInterface row, RowwisePhaseOpInterface first,
                 const CachedValue &cached) {
  Location loc = row->getLoc();
  Type type = cached.value.getType();
  OpBuilder builder = OpBuilder::atBlockBegin(&row->getRegion(0).front());
  auto cache = IterCacheOp::create(builder, loc, type);
  Block *block = builder.createBlock(&cache.getBody(), {},
                                     {first.getIndexArg().getType()}, {loc});
  IRMapping mapping;
  mapping.map(first.getIndexArg(), block->getArgument(0));
  SmallVector<Operation *> ops = exprOps(cached.value, first).takeVector();
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : ops) {
    // A constant the phase reads from the row comes after the cache.
    for (Value operand : op->getOperands())
      if (auto constant = operand.getDefiningOp<ConstantScalarOp>();
          constant && !mapping.contains(operand) &&
          !first->isProperAncestor(constant))
        builder.clone(*constant, mapping);
    builder.clone(*op, mapping);
  }
  YieldOp::create(builder, loc, mapping.lookup(cached.value));

  for (Value copy : cached.copies) {
    Operation *def = copy.getDefiningOp();
    builder.setInsertionPoint(def);
    copy.replaceAllUsesWith(
        IterCacheLoadOp::create(
            builder, copy.getLoc(), type, cache.getResult(),
            def->getParentOfType<RowwisePhaseOpInterface>().getIndexArg())
            .getResult());
  }
}

void eraseDeadOps(RowwisePhaseOpInterface phase) {
  for (Operation &op : llvm::make_early_inc_range(
           llvm::reverse(phase->getRegion(0).front().getOperations())))
    if (isOpTriviallyDead(&op))
      op.erase();
}

void cacheRow(RowwiseOpInterface row) {
  Block &body = row->getRegion(0).front();
  if (!body.getOps<IterCacheOp>().empty())
    return;
  auto phases = llvm::to_vector(body.getOps<RowwisePhaseOpInterface>());
  if (phases.empty() || !isa<IterReduceOp>(phases.front()))
    return;

  // The input the first fold loads first at its own index.
  RowwisePhaseOpInterface first = phases.front();
  Value tensor;
  first->walk([&](LoadOp load) {
    if (load.getIndex() != first.getIndexArg())
      return WalkResult::advance();
    tensor = load.getTensor();
    return WalkResult::interrupt();
  });
  if (!tensor)
    return;

  std::optional<CachedValue> cached = chooseCachedValue(phases, tensor);
  ASSERT_STREAM(cached, << "the load of the reduced input always qualifies");
  createCache(row, first, *cached);
  for (RowwisePhaseOpInterface phase : phases)
    eraseDeadOps(phase);
}

struct ReductionCSEPass : public impl::ReductionCSEBase<ReductionCSEPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([](RowwiseOpInterface row) { cacheRow(row); });
  }
};

} // namespace

} // namespace M::MAP
