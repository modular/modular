//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/TransformUtils/ControlFlowUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

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
  return cast<PointerType>(alloc.getType()).getElementType();
}

/// We can promote a stack allocation if all its uses are as the pointer to
/// loads and stores and no load or store crosses a region of an unknown
/// operation.
static bool canPromote(StackAllocationOp alloc) {
  for (Operation *user : alloc->getUsers()) {
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
      continue;
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

static ErrorOr<DebugInfo::DIExprAttr>
mem2RegLeafConversion(DebugInfo::DIType irType) {
  // Unwrap the DIPointerType if there is one and use the new type.
  DebugInfo::DIType valueType;
  if (auto ptr = dyn_cast<DebugInfo::DIPointerType>(irType))
    valueType = ptr.getElementType();
  else if (auto ptr =
               dyn_cast<DebugInfo::DITargetIndependentPointerType>(irType))
    valueType = ptr.getElementType();
  else if (auto unresolved = dyn_cast<DebugInfo::DIUnresolvedMLIRType>(irType))
    if (auto ptr = dyn_cast<PointerType>(unresolved.getType()))
      valueType = DebugInfo::DIUnresolvedMLIRType::get(ptr.getElementType());

  if (!valueType)
    return Error("Unexpected non-pointer type being unwrapped.");

  auto newIrValue = DebugInfo::DIIRValueExprAttr::get(valueType);
  return DebugInfo::DIRefOfExprAttr::get(newIrValue, irType);
}

namespace {
/// Current state of a promoted stack allocation.
struct PromotedStackAlloc {
  /// The latest value that the alloc'd value has.
  Value currValue;
  /// The attributes that will be used to create DebugInfo ValueOps for the
  /// promoted StackAllocation. One alloc may be mapped to multiple source
  /// variables (each via a DebugInfo ValueOp) after inlining. This map allows
  /// tracking stores to them separately.
  struct DebugValue {
    DebugInfo::DILocalVariableAttr varInfo;
    DebugInfo::DIExprAttr conversionExpr;
  };
  DenseMap<DebugInfo::DISubprogramAttr, DebugValue> debugValues;

  /// Get the current value of this promoted stack allocation. If none exist
  /// yet, create an undef with the same type and return that.
  Value getCurrValueOrUndef(StackAllocationOp alloc, Operation *user) {
    if (LLVM_LIKELY(currValue))
      return currValue;
    // If the value is undefined, materialize an undef operation.
    UndefOp undef =
        OpBuilder(user).create<UndefOp>(user->getLoc(), getAllocType(alloc));
    // Create a DebugInfo ValueOp right after this undef.
    updateValue(undef, undef);
    return undef;
  }

  /// Register that a promoted StackAllocation needs DebugInfo.
  /// The caller pass in the existing DebugInfo ValueOp for the StackAllocation
  /// so that when future stores into this StackAllocation gets transformed, a
  /// DebugInfo ValueOp can be created at the previous store site.
  ErrorOrSuccess
  registerDebugValue(StackAllocationOp alloc, DebugInfo::ValueOp value,
                     DebugInfo::DIExprLeafReplacer &exprLeafReplacer) {
    ErrorOr<DebugInfo::DIExprAttr> newConversionExpr =
        exprLeafReplacer.apply(value.getConversionExprAttr());
    // Not enough source information available to track this transformation.
    // Cannot debug this local variable anymore.
    if (failed(newConversionExpr))
      return success();

    DebugInfo::DISubprogramAttr scope =
        extractPreInlineSubprogramScope(value->getLoc());
    if (!scope)
      return Error(
          "location of debug value does not contain a subprogram scope");
    debugValues[scope] = {value.getValueInfo(), newConversionExpr.get()};
    return success();
  }

  /// Update the current value of this promoted alloc.
  /// Also creates a new DebugInfo ValueOp with this new value if a DebugInfo
  /// ValueOp existed previously for this scope.
  void updateValue(Operation *mutator, Value newValue) {
    currValue = newValue;

    if (debugValues.empty())
      return;

    // Duplicate a DebugInfo::ValueOp for `newValue` if one existed before.
    // The new op is created after `op`.
    // Walk the series of inlined scopes of the mutator op from outermost caller
    // to innermost callee. For each scope where a variable is registered with
    // this value, create a DebugInfo::ValueOp for that variable.
    OpBuilder b(mutator->getContext());
    b.setInsertionPointAfter(mutator);

    // The current location corresponding to the depth of the walk.
    // Since the walk is from caller to callee, the CallSite tree created is
    // caller-side heavy.
    LocationAttr cumulativeLoc;
    // Update `cumulativeLoc` with a new callee location.
    auto appendCallee = [&cumulativeLoc](LocationAttr callee) {
      if (!cumulativeLoc) {
        cumulativeLoc = callee;
        return;
      }
      cumulativeLoc = mlir::CallSiteLoc::get(callee, cumulativeLoc);
    };

    DebugInfo::walkLocation(
        mutator->getLoc(), DebugInfo::LocWalkPolicy::CallerPriority,
        [&](Location loc) -> WalkResult {
          // Update the current cumulative location at each leaf location.
          if (isa<FileLineColLoc>(loc)) {
            appendCallee(loc);
          } else if (auto fused =
                         dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(
                             loc)) {
            appendCallee(loc);

            DebugInfo::DISubprogramAttr mutatorSubprogram =
                DebugInfo::getParentScopeOfType<DebugInfo::DISubprogramAttr>(
                    fused.getMetadata());
            if (auto it = debugValues.find(mutatorSubprogram);
                it != debugValues.end()) {
              DebugValue &dbgValue = it->second;
              b.create<DebugInfo::ValueOp>(cumulativeLoc, newValue,
                                           dbgValue.varInfo,
                                           dbgValue.conversionExpr);
            }
            return WalkResult::skip();
          }
          return WalkResult::advance();
        });
  }

private:
  /// Get the innermost scope from a series of callsite locations.
  static DebugInfo::DISubprogramAttr
  extractPreInlineSubprogramScope(Location loc) {
    return DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
        loc, DebugInfo::LocWalkPolicy::CalleePriority);
  }
};
} // namespace

static LogicalResult
processRegion(Region &region, const HLCF::CFGAnalysis &cfg,
              llvm::MapVector<StackAllocationOp, PromotedStackAlloc> &state,
              DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
                  &termVariants,
              DebugInfo::DIExprLeafReplacer &exprLeafReplacer,
              PassStats &stats) {
  // This analysis only works on single-block regions.
  if (!llvm::hasSingleElement(region)) {
    return region.getParentOp()->emitError(
        "'mem-2-reg' can only be run on operations with all single block "
        "regions");
  }

  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    if (auto alloc = dyn_cast<StackAllocationOp>(op)) {
      // If we can promote this stack allocation, initialize its state with an
      // undefined value.
      if (canPromote(alloc))
        state.try_emplace(alloc);
      continue;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      // If we can elide this load, replace the result of the load with the last
      // value of the stack allocation.
      if (auto alloc = load.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          load.replaceAllUsesWith(it->second.getCurrValueOrUndef(alloc, load));
          load.erase();
          ++stats.numLoadsElided;
        }
      }
      continue;
    }
    if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
      // Delete stale debuginfo for the old stack allocation op.
      if (auto alloc = value.getValue().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          auto newValue =
              it->second.registerDebugValue(alloc, value, exprLeafReplacer);
          if (failed(newValue))
            return value.emitError() << newValue.getError();
          value.erase();
        }
      }
      continue;
    }
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(op)) {
      assert(op.getNumOperands() == 1);
      if (state.contains(op.getOperand(0).getDefiningOp<StackAllocationOp>()))
        op.erase();
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      // If we can elide this store, capture the last written value and erase
      // the operation.
      if (auto alloc = store.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (auto it = state.find(alloc); it != state.end()) {
          it->second.updateValue(store, store.getArg());
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
            state.find(alloc)->second.getCurrValueOrUndef(alloc, &op));
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
        if (failed(processRegion(region, cfg, state, termVariants,
                                 exprLeafReplacer, stats)))
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
      llvm::MapVector<StackAllocationOp, PromotedStackAlloc> nestedState =
          state;

      if (!variant.empty()) {
        // Determine if there are any region predecessors.
        if (regionPreds[region.getRegionNumber()]) {
          for (auto [i, alloc] : llvm::enumerate(variant)) {
            Type allocType = getAllocType(alloc);
            // Bind the block argument to the value of the variant allocation.
            nestedState[alloc] = {
                node.insertArgumentToRegion(op.getLoc(), allocType, i, region),
                state.find(alloc)->second.debugValues};
          }
          // If one of the predecessors is the parent operation, we need to
          // add initializer operands to it if this hasn't already been done.
          if (!parentHasInit && parentPred[region.getRegionNumber()]) {
            parentHasInit = true;
            SmallVector<Value> initOperands;
            for (StackAllocationOp alloc : variant) {
              initOperands.push_back(
                  state.find(alloc)->second.getCurrValueOrUndef(alloc, &op));
            }

            node.insertVariants(initOperands);
          }
        }
      }
      // Okay, now recurse into the region.
      if (failed(processRegion(region, cfg, nestedState, termVariants,
                               exprLeafReplacer, stats)))
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
      for (auto [i, alloc] : llvm::enumerate(variant)) {
        // Update currValue without creating a new debug value, since the
        // mutator inside the nested scope will have noted when the value was
        // updated.
        state.find(alloc)->second.currValue = newOp->getResult(iterStart + i);
      }
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
  llvm::MapVector<StackAllocationOp, PromotedStackAlloc> entryState;
  DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
      termVariants;
  for (Region &region : getOperation()->getRegions()) {
    // Reuse the same memory for the maps each time.
    entryState.clear();
    termVariants.clear();
    DebugInfo::DIExprLeafReplacer exprLeafReplacer(mem2RegLeafConversion);
    if (failed(processRegion(region, cfg, entryState, termVariants,
                             exprLeafReplacer, stats)))
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
