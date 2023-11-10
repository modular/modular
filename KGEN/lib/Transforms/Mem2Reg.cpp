//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ControlFlowUtils.h"
#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_MEM2REG
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Statistics collected for this pass.
struct PassStats {
  unsigned numAllocsElided = 0;
  unsigned numLoadsElided = 0;
  unsigned numStoresElided = 0;
};
} // namespace

/// Return the pointer element type of an allocation.
static Type getAllocType(StackAllocationOp alloc) {
  return ParamRefType::get(cast<PointerType>(alloc.getType()).getElementType());
}

/// We can promote a stack allocation if all its uses are as the pointer to
/// loads and stores and no load or store crosses a region of an unknown
/// operation.
static bool canPromote(StackAllocationOp alloc) {
  for (Operation *user : alloc->getUsers()) {
    if (isa<LoadOp, DebugInfo::ValueOp>(user)) {
      if (userCrossesFunctionCFG(alloc, user))
        return false;
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getArg() == alloc ||
        userCrossesFunctionCFG(alloc, store))
      return false;
  }
  return true;
}

/// Given a type in a DILocalVariableAttr, unwrap one level of
/// KGEN::PointerType or DIPointerType. This does not perform any other
/// replacements.
/// TODO(#23914): Track this optimization with DWARF expressions.
static DebugInfo::DILocalVariableAttr
unwrapPointer(DebugInfo::DILocalVariableAttr diVarAttr) {
  DebugInfo::DIType type = diVarAttr.getType();
  DebugInfo::DIType newType = type;

  // Unwrap the DIUnresolvedMLIRType (if there is one) and return the new type.
  if (auto unresolved = dyn_cast<DebugInfo::DIUnresolvedMLIRType>(type))
    if (auto ptr = dyn_cast<KGEN::PointerType>(unresolved.getType()))
      newType = DebugInfo::DIUnresolvedMLIRType::get(ptr.getElementAsType());

  // Unwrap the DIPointerType if there is one and use the new type.
  if (auto ptr = dyn_cast<DebugInfo::DIPointerType>(type))
    newType = ptr.getElementType();
  else if (auto ptr = dyn_cast<DebugInfo::DITargetIndependentPointerType>(type))
    newType = ptr.getElementType();

  return DebugInfo::DILocalVariableAttr::get(
      diVarAttr.getScope(), diVarAttr.getName(), diVarAttr.getFile(),
      diVarAttr.getLine(), diVarAttr.getArg(), newType.getAlignInBits(),
      newType);
}

static LogicalResult
processRegion(Region &region, const HLCF::CFGAnalysis &cfg,
              llvm::MapVector<StackAllocationOp, Value> &state,
              DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
                  &termVariants,
              PassStats &stats) {
  // This analysis only works on single-block regions.
  if (!llvm::hasSingleElement(region)) {
    return region.getParentOp()->emitError(
        "'mem-2-reg' can only be run on operations with all single block "
        "regions");
  }

  auto valueOrUndef = [](StackAllocationOp alloc, Operation *op,
                         Value value) -> Value {
    if (LLVM_LIKELY(value))
      return value;
    // If the value is undefined, materialize an undef operation.
    return OpBuilder(op).create<UndefOp>(op->getLoc(), getAllocType(alloc));
  };

  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    if (auto alloc = dyn_cast<StackAllocationOp>(op)) {
      // If we can promote this stack allocation, initialize its state with an
      // undefined value.
      if (canPromote(alloc))
        state.insert({alloc, {}});
      continue;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      // If we can elide this load, replace the result of the load with the last
      // value of the stack allocation.
      if (auto alloc = load.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          load.replaceAllUsesWith(valueOrUndef(alloc, load, it->second));
          load.erase();
          ++stats.numLoadsElided;
        }
      }
      continue;
    }
    if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
      // Replace the variable with its current value.
      if (auto alloc = value.getValue().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          OpBuilder b(value);
          b.create<DebugInfo::ValueOp>(value.getLoc(),
                                       valueOrUndef(alloc, value, it->second),
                                       unwrapPointer(value.getValueInfo()));
          value.erase();
        }
      }
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      // If we can elide this store, capture the last written value and erase
      // the operation.
      if (auto alloc = store.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          it->second = store.getArg();
          store.erase();
          ++stats.numStoresElided;
        }
      }
      continue;
    }
    if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(op)) {
      // Look up the required variant values, if there are any.
      auto it = termVariants.find(term);
      if (it == termVariants.end() || it->second.empty())
        continue;
      // Bind the last values to the operands.
      SmallVector<Value> newOperands;
      for (StackAllocationOp alloc : it->second) {
        newOperands.push_back(
            valueOrUndef(alloc, &op, state.find(alloc)->second));
      }
      term.insertVariants(newOperands);
      continue;
    }

    // If this operation has regions, recurse into the regions.
    unsigned numRegions = op.getNumRegions();
    if (!numRegions)
      continue;

    auto node = dyn_cast<HLCF::ControlFlowNode>(op);
    if (!node) {
      // This is an unknown operation. Process it as if it were isolated.
      for (Region &region : op.getRegions())
        if (failed(processRegion(region, cfg, state, termVariants, stats)))
          return failure();
      continue;
    }

    // For control-flow operations, all current stack allocations are visible
    // within the regions. Determine which are variant. These values will have
    // to be carried through the regions using iteration variables.
    std::vector<StackAllocationOp> variant;
    for (StackAllocationOp alloc : llvm::make_first_range(state)) {
      for (Operation *user : alloc->getUsers()) {
        auto store = dyn_cast<StoreOp>(user);
        if (!store)
          continue;
        if (op.isProperAncestor(store)) {
          variant.push_back(alloc);
          break;
        }
      }
    }

    // Map the required variant values to predecessor terminators of the end of
    // the operation and to each region.
    llvm::BitVector regionPreds(op.getNumRegions());
    llvm::BitVector parentPred(op.getNumRegions());
    if (!variant.empty()) {
      for (Operation *pred : cfg.predecessors.at({node, {}})) {
        if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(pred))
          termVariants.try_emplace(term, variant);
      }

      for (Region &region : op.getRegions()) {
        ArrayRef<Operation *> preds =
            cfg.predecessors.at({node, region.getRegionNumber()});
        for (Operation *pred : preds) {
          if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(pred)) {
            termVariants.try_emplace(term, variant);
            regionPreds.set(region.getRegionNumber());
          } else {
            assert(pred == &op);
            parentPred.set(region.getRegionNumber());
          }
        }
      }
    }

    // For each region with region predecessors (demarcated by a terminator)
    // and variant allocations, introduce block arguments.
    bool parentHasInit = false;
    for (Region &region : op.getRegions()) {
      // Copy the current state.
      llvm::MapVector<StackAllocationOp, Value> nestedState = state;

      if (!variant.empty()) {
        // Determine if there are any region predecessors.
        if (regionPreds[region.getRegionNumber()]) {
          for (auto [i, alloc] : llvm::enumerate(variant)) {
            Type allocType = getAllocType(alloc);
            // Bind the block argument to the value of the variant allocation.
            nestedState[alloc] =
                node.insertArgumentToRegion(op.getLoc(), allocType, i, region);
          }
          // If one of the predecessors is the parent operation, we need to
          // add initializer operands to it if this hasn't already been done.
          if (!parentHasInit && parentPred[region.getRegionNumber()]) {
            parentHasInit = true;
            SmallVector<Value> initOperands;
            for (StackAllocationOp alloc : variant) {
              initOperands.push_back(
                  valueOrUndef(alloc, &op, state.find(alloc)->second));
            }

            node.insertVariants(initOperands);
          }
        }
      }
      // Okay, now recurse into the region.
      if (failed(processRegion(region, cfg, nestedState, termVariants, stats)))
        return failure();

      // Erase elided allocations in the nested region.
      for (StackAllocationOp alloc : llvm::make_first_range(nestedState)) {
        if (!state.contains(alloc)) {
          alloc.erase();
          ++stats.numAllocsElided;
        }
      }
    }

    // After processing the regions, we need to add results to the operation
    // to merge the values of variant allocations, and then bind those as the
    // current values of those allocations.
    if (!variant.empty()) {
      if (!op.hasTrait<OpTrait::VariadicResults>()) {
        return op.emitOpError(
            "must have trailing variadic results to be used in 'mem-2-reg'");
      }
      SmallVector<Type> newTypes = llvm::to_vector(op.getResultTypes());
      for (StackAllocationOp alloc : variant)
        newTypes.push_back(getAllocType(alloc));
      Operation *newOp = Operation::create(
          op.getLoc(), op.getName(), newTypes, op.getOperands(),
          op.getAttrDictionary(), nullptr, {}, op.getNumRegions());
      OpBuilder(&op).insert(newOp);
      for (unsigned i = 0, e = op.getNumRegions(); i != e; ++i)
        newOp->getRegion(i).takeBody(op.getRegion(i));
      unsigned iterStart = op.getNumResults();
      for (auto [i, alloc] : llvm::enumerate(variant))
        state.find(alloc)->second = newOp->getResult(iterStart + i);
      op.replaceAllUsesWith(newOp->getResults().slice(0, iterStart));
      op.erase();
    }
  }

  return success();
}

namespace {
struct Mem2RegPass : public M::KGEN::impl::Mem2RegBase<Mem2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Mem2RegPass::runOnOperation() {
  auto &cfg = getAnalysis<HLCF::CFGAnalysis>();
  PassStats stats;
  for (Region &region : getOperation()->getRegions()) {
    llvm::MapVector<StackAllocationOp, Value> entryState;
    DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
        termVariants;
    if (failed(processRegion(region, cfg, entryState, termVariants, stats)))
      return signalPassFailure();
    // Erase elided allocations.
    for (StackAllocationOp alloc : llvm::make_first_range(entryState)) {
      alloc.erase();
      ++stats.numAllocsElided;
    }
  }

  numAllocsElided = stats.numAllocsElided;
  numLoadsElided = stats.numLoadsElided;
  numStoresElided = stats.numStoresElided;

  // Control-flow is not modified.
  markAnalysesPreserved<HLCF::CFGAnalysis>();
}
