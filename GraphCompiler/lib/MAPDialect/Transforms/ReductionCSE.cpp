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
// element in a `map.iter.tile`, the tile the Row loads, and each phase reads it
// back instead of computing it again:
//
//   map.iter.reduce index (%i) {          %c = map.iter.tile index (%k) {
//     %v = mul(%x[%i], %s[%i])              %v = mul(%x[%k], %s[%k])
//     ...                           =>      map.yield %v
//   }                                     }
//   map.iter.map index (%j) {             map.iter.reduce index (%i) {
//     %v = mul(%x[%j], %s[%j])              %v = map.iter.tile.load %c[%i]
//     ...                                   ...
//   }                                     }
//                                         map.iter.map index (%j) {
//                                           %v = map.iter.tile.load %c[%j]
//                                           ...
//
// First, a value the map phase computes at its element and again at another
// element of the row, in a concat select's regions, is computed once per
// element in a shared `map.iter.cache`. The regions read the row the cache
// keeps, `offset` elements away (rotate-half reads each element's partner):
//
//   map.iter.map index (%j) {             %c = map.iter.cache index (%i) {
//     %n = f(%x[%j])                        %n = f(%x[%i])
//     select(%j) {                  =>      map.yield %n
//       (%a) { f(%x[slice(%a)]) }         } {shared = true}
//     } ...                               map.iter.map index (%j) {
//   }                                       %n = map.iter.cache.load %c[%j]
//                                           select(%j) {
//                                             (%a) { map.iter.cache.load
//                                                      %c[%j] offset(64) }
//                                           } ...
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

// Whether `phase`, once it reads `copies` from the tile, no longer loads
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

struct TileValue {
  // An element expression of the first fold.
  Value value;
  // Its copies in every phase, `value` included.
  SmallVector<Value> copies;
};

// The row's tile. Of the first fold's element expressions over
// `tensor` that every phase computes wherever it reads `tensor`, take the one
// with the narrowest type, then the most ops: a wider one would widen the
// tile, a smaller one would recompute work in every phase. The load itself
// always qualifies.
std::optional<TileValue> chooseTile(ArrayRef<RowwisePhaseOpInterface> phases,
                                    Value tensor) {
  RowwisePhaseOpInterface first = phases.front();
  ElementExprs exprs;
  std::optional<TileValue> best;
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
      TileValue candidate{value, {}};
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

// Computes `tile.value` in a `map.iter.tile` at the start of `row`, and
// makes each copy read it.
void createTile(RowwiseOpInterface row, RowwisePhaseOpInterface first,
                const TileValue &tile) {
  Location loc = row->getLoc();
  Type type = tile.value.getType();
  OpBuilder builder = OpBuilder::atBlockBegin(&row->getRegion(0).front());
  auto tileOp = IterTileOp::create(builder, loc, type);
  Block *block = builder.createBlock(&tileOp.getBody(), {},
                                     {first.getIndexArg().getType()}, {loc});
  IRMapping mapping;
  mapping.map(first.getIndexArg(), block->getArgument(0));
  SmallVector<Operation *> ops = exprOps(tile.value, first).takeVector();
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : ops) {
    // A constant the phase reads from the row comes after the tile.
    for (Value operand : op->getOperands())
      if (auto constant = operand.getDefiningOp<ConstantScalarOp>();
          constant && !mapping.contains(operand) &&
          !first->isProperAncestor(constant))
        builder.clone(*constant, mapping);
    builder.clone(*op, mapping);
  }
  YieldOp::create(builder, loc, mapping.lookup(tile.value));

  for (Value copy : tile.copies) {
    Operation *def = copy.getDefiningOp();
    builder.setInsertionPoint(def);
    copy.replaceAllUsesWith(
        IterTileLoadOp::create(
            builder, copy.getLoc(), type, tileOp.getResult(),
            def->getParentOfType<RowwisePhaseOpInterface>().getIndexArg())
            .getResult());
  }
}

// Erases the ops of `region` and its nested regions that nothing reads.
void eraseDeadOps(Region &region) {
  for (Operation &op : llvm::make_early_inc_range(
           llvm::reverse(region.front().getOperations()))) {
    for (Region &inner : op.getRegions())
      eraseDeadOps(inner);
    if (isOpTriviallyDead(&op))
      op.erase();
  }
}

void createRowTile(RowwiseOpInterface row) {
  Block &body = row->getRegion(0).front();
  if (!body.getOps<IterTileOp>().empty())
    return;
  // A phase that reads a cache gets the cache, not the tile, so it keeps its
  // own loads of the input.
  SmallVector<RowwisePhaseOpInterface> phases;
  for (RowwisePhaseOpInterface phase : body.getOps<RowwisePhaseOpInterface>())
    if (!phase.getReadCache())
      phases.push_back(phase);
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

  std::optional<TileValue> tile = chooseTile(phases, tensor);
  ASSERT_STREAM(tile, << "the load of the reduced input always qualifies");
  createTile(row, first, *tile);
  for (RowwisePhaseOpInterface phase : phases)
    eraseDeadOps(phase->getRegion(0));
}

// How far `index` moves from `phaseIndex` along `axis`, the other coordinates
// kept: through a concat select's region on `axis`, which runs at its index
// minus the extents before it, and slices that move only `axis`.
std::optional<int64_t> shiftAlong(Value index, Value phaseIndex, int64_t axis) {
  if (index == phaseIndex)
    return 0;
  if (auto arg = dyn_cast<BlockArgument>(index)) {
    auto select = dyn_cast<ConcatSelectOp>(arg.getOwner()->getParentOp());
    if (!select || select.getSpec().getAxis() != axis)
      return std::nullopt;
    std::optional<int64_t> shift =
        shiftAlong(select.getIndex(), phaseIndex, axis);
    unsigned region = arg.getOwner()->getParent()->getRegionNumber();
    for (MOSH::ShapeAttr input :
         select.getSpec().getInputShapes().take_front(region)) {
      std::optional<int64_t> extent =
          MOSH::ShapeAttr::getStaticDim(input[axis]);
      if (!shift || !extent)
        return std::nullopt;
      *shift -= *extent;
    }
    return shift;
  }
  auto slice = index.getDefiningOp<SliceIndexOp>();
  if (!slice)
    return std::nullopt;
  std::optional<int64_t> shift = shiftAlong(slice.getIndex(), phaseIndex, axis);
  SliceSpecAttr spec = slice.getSpec();
  for (auto [d, start, step] : llvm::enumerate(spec.getStarts().asArrayRef(),
                                               spec.getSteps().asArrayRef()))
    if (!shift || step != 1 || ShapedType::isDynamic(start) ||
        (static_cast<int64_t>(d) != axis && start != 0))
      return std::nullopt;
  return *shift + spec.getStarts()[axis];
}

struct MovedCopy {
  Value value;
  int64_t offset;
};

// A value a map phase computes at its element from a load, and its copies at
// other elements of the row.
struct SharedValue {
  Value value;
  SmallVector<MovedCopy> copies;
};

// Values `phase` computes from a load, without nested regions, so a cache
// can compute them instead.
SmallVector<Value> loadedValues(IterMapOp phase) {
  auto rowPhase = cast<RowwisePhaseOpInterface>(*phase);
  SmallVector<Value> values;
  for (Operation &op : phase.getBody().front()) {
    if (op.getNumResults() != 1 || !op.getResult(0).getType().isIntOrFloat())
      continue;
    llvm::SetVector<Operation *> ops = exprOps(op.getResult(0), rowPhase);
    if (llvm::any_of(ops, llvm::IsaPred<LoadOp>) &&
        llvm::none_of(ops, [](Operation *o) { return o->getNumRegions(); }))
      values.push_back(op.getResult(0));
  }
  return values;
}

// The value of `phase` whose copies its concat selects on the row's axis
// compute at other elements of the row, in their regions. The outermost copy
// counts, and all copies must be of one value: a row has one cache.
std::optional<SharedValue> findSharedValue(IterRowwiseOp row, IterMapOp phase) {
  auto rowPhase = cast<RowwisePhaseOpInterface>(*phase);
  Value index = rowPhase.getIndexArg();
  SmallVector<Value> candidates = loadedValues(phase);
  std::optional<SharedValue> shared;
  bool conflict = false;
  // Records `copy` when it is a candidate at an index moved along the row. A
  // copy that reads a value the phase computes at its own element isn't one.
  auto matchCopy = [&](Value copy, Region &region) {
    llvm::SetVector<Operation *> ops = exprOps(copy, rowPhase);
    if (llvm::any_of(ops, [&](Operation *def) {
          return llvm::any_of(def->getOperands(), [&](Value operand) {
            return rowPhase->isAncestor(
                       operand.getParentRegion()->getParentOp()) &&
                   !region.isAncestor(operand.getParentRegion());
          });
        }))
      return false;
    for (Operation *def : ops)
      for (Value moved : def->getOperands()) {
        std::optional<int64_t> offset =
            isa<IndexType>(moved.getType())
                ? shiftAlong(moved, index, row.getAxis())
                : std::nullopt;
        if (!offset || *offset == 0)
          continue;
        ElementExprs exprs;
        for (Value value : candidates) {
          if (!exprs.same(copy, moved, value, index))
            continue;
          conflict |= shared && shared->value != value;
          if (!shared)
            shared = SharedValue{value, {}};
          shared->copies.push_back({copy, *offset});
          return true;
        }
      }
    return false;
  };
  phase.walk([&](ConcatSelectOp select) {
    if (select.getIndex() != index ||
        select.getSpec().getAxis() != static_cast<int64_t>(row.getAxis()))
      return;
    for (Region &region : select.getInputs()) {
      SmallVector<Operation *> ops;
      region.walk([&](Operation *op) { ops.push_back(op); });
      llvm::DenseSet<Operation *> covered;
      for (Operation *op : llvm::reverse(ops))
        if (!covered.contains(op) && op->getNumResults() == 1 &&
            matchCopy(op->getResult(0), region)) {
          llvm::SetVector<Operation *> inner =
              exprOps(op->getResult(0), rowPhase);
          covered.insert(inner.begin(), inner.end());
        }
    }
  });
  if (conflict)
    return std::nullopt;
  return shared;
}

// Computes `shared.value` in a shared cache before `phase`, which reads it
// at its element, and its copies further along the row.
void createSharedCache(IterMapOp phase, const SharedValue &shared) {
  Location loc = phase.getLoc();
  Value index = cast<RowwisePhaseOpInterface>(*phase).getIndexArg();
  Type type = shared.value.getType();
  OpBuilder builder(phase);
  auto cache = IterCacheOp::create(builder, loc, type, /*shared=*/true);
  Block *block =
      builder.createBlock(&cache.getBody(), {}, {index.getType()}, {loc});
  IRMapping mapping;
  mapping.map(index, block->getArgument(0));
  SmallVector<Operation *> ops =
      exprOps(shared.value, cast<RowwisePhaseOpInterface>(*phase)).takeVector();
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : ops)
    builder.clone(*op, mapping);
  YieldOp::create(builder, loc, mapping.lookup(shared.value));

  auto readAt = [&](Value value, int64_t offset) {
    builder.setInsertionPoint(value.getDefiningOp());
    value.replaceAllUsesWith(IterCacheLoadOp::create(builder, value.getLoc(),
                                                     type, cache.getResult(),
                                                     index, offset)
                                 .getResult());
  };
  readAt(shared.value, 0);
  for (const MovedCopy &copy : shared.copies)
    readAt(copy.value, copy.offset);
  eraseDeadOps(phase.getBody());
}

void cacheSharedValue(RowwiseOpInterface rowOp) {
  auto row = dyn_cast<IterRowwiseOp>(*rowOp);
  Block &body = rowOp->getRegion(0).front();
  // A row has one cache, which follows a fold and comes before the tile.
  if (!row || !body.getOps<IterCacheOp>().empty() ||
      !body.getOps<IterTileOp>().empty() ||
      body.getOps<IterReduceOp>().empty() || body.getOps<IterMapOp>().empty())
    return;
  IterMapOp phase;
  for (IterMapOp map : body.getOps<IterMapOp>())
    phase = map;
  if (std::optional<SharedValue> shared = findSharedValue(row, phase))
    createSharedCache(phase, *shared);
}

struct ReductionCSEPass : public impl::ReductionCSEBase<ReductionCSEPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([](RowwiseOpInterface row) {
      cacheSharedValue(row);
      createRowTile(row);
    });
  }
};

} // namespace

} // namespace M::MAP
