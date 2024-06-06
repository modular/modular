//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TargetAdapter.h"
#include "NVPTXAdapter.h"

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace M::DebugInfo;

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// TargetAdapter
//===----------------------------------------------------------------------===//
TargetAdapter DebugInfo::getTargetAdapter(M::TargetInfoAttr target) {
  if (target && target.getTriple().isNVPTX())
    return getNVPTXAdapter();
  return getFallbackAdapter();
}

TargetAdapter DebugInfo::getFallbackAdapter() {
  return TargetAdapter{populateFallbackConversionPatterns,
                       [](ModuleOp module) { sinkDebugKills(module); },
                       convertDbgValueToDeclare};
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConvertLineTableLocOp : public OpRewritePattern<LineTableLocOp> {
  ConvertLineTableLocOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<LineTableLocOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(LineTableLocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<LLVM::InlineAsmOp>(
        replacer.replace<LocationAttr>(op.getLoc()), TypeRange{}, ValueRange{},
        "nop", "", /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr());
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

void DebugInfo::populateFallbackConversionPatterns(
    DIAttrTypeReplacer &replacer, RewritePatternSet &patterns) {
  patterns.add<ConvertLineTableLocOp>(patterns.getContext(), replacer);
}

//===----------------------------------------------------------------------===//
// sinkDebugKills
//===----------------------------------------------------------------------===//
namespace {
/// A linearized canonical representation of an inline call stack (as opposed to
/// a binary-tree-based representation used by CallSiteLoc) that allows easy
/// ancestor-child comparison.
class CallStack {
public:
  CallStack() = default;
  CallStack(Location overallLoc) {
    walkLocation(overallLoc, LocWalkPolicy::CallerPriority, [&](Location loc) {
      if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc)) {
        if (fusedLoc.getLocations().size() != 1)
          return WalkResult::advance();

        if (auto fileLineCol =
                dyn_cast<FileLineColLoc>(fusedLoc.getLocations().front()))
          frames.emplace_back(fusedLoc.getMetadata(), fileLineCol.getLine());
      }
      return WalkResult::advance();
    });
  }

  /// The call stack ordered from caller to callee.
  /// Each frame encodes the scope of the location & the line number.
  using Frame = std::pair<DIScopeAttr, unsigned>;
  SmallVector<Frame> frames;
};

/// Maps each frame of a CallStack to some user-defined data `T`.
template <typename T>
class CallStackWith {
public:
  bool empty() const { return dataFrames.empty(); }

  /// Reference to the data value mapped to the last (innermost) frame.
  T &backData() { return dataFrames.back().second; }

  /// Update the internal call stack to represent `newStack` instead.
  /// Any stack frame that will no longer exist is considered invalidated, and
  /// will be returned in the order of their positions in the call stack.
  /// Each newly added stack frame will come with a default-constructed `T`.
  ///
  /// For example, calling with
  ///   dataFrames = [(L0, T0), (L1, T1), (L2, T2), (L3, T3)]
  ///   newStack   = [L0, L1, L4, L5]
  /// results in
  ///   dataFrames = [(L0, T0), (L1, T1), (L4, T()), (L5, T())]
  /// and returns [T2, T3].
  SmallVector<T> updateTo(const CallStack &newStack) {
    // Walk until `newStack.frames` & `dataFrames` diverge.
    auto thisIter = dataFrames.begin();
    auto newIter = newStack.frames.begin();
    for (; thisIter != dataFrames.end() && newIter != newStack.frames.end();
         ++thisIter, ++newIter)
      if (thisIter->first != *newIter)
        break;

    SmallVector<T> invalidated;
    // Diverged in the middle of `dataFrames`. Invalidate everything afterwards.
    if (thisIter != dataFrames.end()) {
      std::transform(thisIter, dataFrames.end(),
                     std::back_inserter(invalidated),
                     [](auto it) { return it.second; });
      dataFrames.truncate(dataFrames.size() - invalidated.size());
    }

    // Append anything at or after `newIter` to `dataFrames`.
    for (; newIter != newStack.frames.end(); ++newIter)
      dataFrames.emplace_back(*newIter, T());

    return invalidated;
  }

private:
  /// The call stack ordered from caller to callee.
  /// Each frame encodes both a location and a custom data `T`.
  SmallVector<std::pair<CallStack::Frame, T>> dataFrames;
};
} // namespace

/// Sink kill Debug Value ops so that they are the last instructions from
/// their source line. This way variables are guaranteed to be killed only at
/// the end of the line.
void DebugInfo::sinkDebugKills(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // The kill Debug Value Ops corresponding to the current line at each
      // inlined scope.
      CallStackWith<SmallVector<KillOp>> pendingKillsByLoc;
      // The debug kills that have been superceded by a non-kill debug value.
      // This ensures we never change the order of values for a variable.
      DenseSet<DILocalVariableAttr> staleKills;
      for (Operation &op : llvm::make_early_inc_range(block.getOperations())) {
        if (auto value = dyn_cast<ValueOp>(op))
          staleKills.insert(value.getValueInfo());

        // Ops without a location follow the location of the previous op.
        const CallStack callStack(op.getLoc());
        if (!callStack.frames.empty()) {
          SmallVector<SmallVector<KillOp>> invalidated =
              pendingKillsByLoc.updateTo(callStack);
          // This is the start of a new line. Move all pending kill debug values
          // before this op.
          for (SmallVector<KillOp> &kills : invalidated) {
            for (KillOp kill : kills) {
              if (!staleKills.contains(kill.getValueInfo()))
                kill->moveBefore(&op);
              else
                kill->erase();
            }
          }
        }

        if (auto kill = dyn_cast<KillOp>(op)) {
          staleKills.erase(kill.getValueInfo());
          if (!pendingKillsByLoc.empty())
            pendingKillsByLoc.backData().push_back(kill);
        }
        sinkDebugKills(&op);
      }

      // Any still pending kills can be moved before the last op of the block.
      if (!pendingKillsByLoc.empty()) {
        Operation *lastOp = &block.back();
        SmallVector<SmallVector<KillOp>> invalidated =
            pendingKillsByLoc.updateTo({});
        for (SmallVector<KillOp> &kills : invalidated) {
          for (KillOp kill : kills) {
            if (!staleKills.contains(kill.getValueInfo()))
              kill->moveBefore(lastOp);
            else
              kill->erase();
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// convertDbgValueToDeclare
//===----------------------------------------------------------------------===//
namespace {
/// Summary of all the variables that are tracked with debug info in a function.
///
/// - A "variable" refers to a source variable (described by the
///   DILocalVariableAttr of a DbgValueOp)
/// - A "value" refers to an IR Value that is used to define the value of a
///   variable at some program location (as an operand of a DbgValueOp).
struct DebugVariableSummary {
public:
  /// Debug info for variables that can be processed by
  /// convertDbgValueToDeclare.
  struct ProcessableVariable {
    ProcessableVariable(LLVM::DbgValueOp op, bool isUndef) {
      if (isUndef)
        this->primaryUndef = op;
      else
        this->valueOps.push_back(op);
    }

    /// The list of dbg.value ops for the variable.
    SmallVector<LLVM::DbgValueOp> valueOps;
    /// In the case of an undef before the first non-undef, that means that the
    /// value starts undefined, and should be treated differently than later
    /// undefs.
    std::optional<LLVM::DbgValueOp> primaryUndef;
    /// Additional undefs indicate that the lifetime of the variable does not
    /// last the entire scope.
    SmallVector<LLVM::DbgValueOp> additionalUndefs;
    /// Whether any dbg.value ops include DW_OP_LLVM_fragment paths.
    bool anyFragments = false;
    /// Whether any dbg.value ops include DW_OP_deref paths.
    bool anyPointers = false;
    /// Whether the deref paths are the same length (IE same number of pointer
    /// dereferences to get to the variable).
    bool allPointersMatchedLength = true;
    /// Whether to skip processing, IE it is not actually processable due to
    /// having an exprLocation that we can't handle.
    bool skip = false;
  };

  llvm::MapVector<LLVM::DILocalVariableAttr, ProcessableVariable>
      processableVariableMap;
};
} // namespace

/// Filter out the debug values that are not needed, and summarize the rest in
/// a DebugVariableSummary.
static DebugVariableSummary
filterAndSummarizeDebugVariables(mlir::FunctionOpInterface func) {
  // Summarize debug values by variable.
  llvm::MapVector<LLVM::DILocalVariableAttr,
                  DebugVariableSummary::ProcessableVariable>
      debugValuesToProcess;
  func->walk([&](LLVM::DbgValueOp op) {
    Value value = op.getValue();
    // Don't build debug info for token values.
    if (isa<LLVM::LLVMTokenType>(value.getType())) {
      op->erase();
      return;
    }

    if (value.getDefiningOp<LLVM::UndefOp>()) {
      auto [iter, inserted] =
          debugValuesToProcess.try_emplace(op.getVarInfo(), op, true);
      if (!inserted)
        iter->second.additionalUndefs.push_back(op);
      return;
    }

    // Not undef.
    auto [iter, inserted] =
        debugValuesToProcess.try_emplace(op.getVarInfo(), op, false);
    if (!inserted)
      iter->second.valueOps.push_back(op);
  });

  for (auto &[var, processable] : debugValuesToProcess) {
    // Check that the locationExpr is well formed.  We can handle locationExprs
    // that are 0 or more DW_OP_deref followed by zero or more
    // DW_OP_LLVM_fragment, but no fragments before deref, and no other
    // operations.
    bool onFirstPass = true;
    uint64_t firstNumPointers = 0;
    for (auto valueOp : processable.valueOps) {
      if (processable.skip)
        break;
      bool foundFragment = false;
      uint64_t numPointers = 0;
      for (auto exprOp : valueOp.getLocationExpr().getOperations()) {
        if (exprOp.getOpcode() == llvm::dwarf::DW_OP_deref) {
          processable.anyPointers = true;
          ++numPointers;
          if (foundFragment) {
            processable.skip = true;
            break;
          }
        } else if (exprOp.getOpcode() == llvm::dwarf::DW_OP_LLVM_fragment) {
          foundFragment = true;
          processable.anyFragments = true;
        } else {
          processable.skip = true;
          break;
        }
      }
      if (onFirstPass)
        firstNumPointers = numPointers;
      else if (firstNumPointers != numPointers)
        processable.allPointersMatchedLength = false;
      onFirstPass = false;
    }
  }

  DebugVariableSummary summary;
  for (auto &[var, processable] : debugValuesToProcess) {
    if (processable.skip)
      continue;
    summary.processableVariableMap.insert({var, processable});
  }
  return summary;
}

void DebugInfo::convertDbgValueToDeclare(ModuleOp module) {
  // A lot more logic is required to make this reverse-mem2reg work when
  // multiple DbgValueOps for one variable exists. Going with the simplest
  // solution for now until we decide to retire this altogether.
  for (auto func : module.getOps<mlir::FunctionOpInterface>()) {
    DebugVariableSummary debugVariableSummary =
        filterAndSummarizeDebugVariables(func);

    for (auto &[varInfo, processable] :
         debugVariableSummary.processableVariableMap) {
      LLVM::DbgValueOp primaryOp = processable.valueOps.empty()
                                       ? processable.primaryUndef.value()
                                       : processable.valueOps[0];
      Value primaryValue = primaryOp.getValue();
      // Don't build debug information for simple constants.
      if (primaryValue.getDefiningOp<LLVM::ConstantOp>() &&
          isa<IntegerType, FloatType>(primaryValue.getType()))
        continue;

      bool anyMutable = processable.valueOps.size() > 1;

      // The converted alloca op that will hold the value if it is converted to
      // a stack allocation.  In the mutable case each processable needs its
      // own, and map each processable's alloca to its primaryValue.
      LLVM::AllocaOp allocaOp;
      llvm::MapVector<LLVM::DbgValueOp, LLVM::AllocaOp> allocaMap;

      // Get the allocaOp for this value. If one has not already been created,
      // create one and save it for the next invocation.
      auto getAllocaOp =
          [&, targetValue = primaryValue](
              DebugVariableSummary::ProcessableVariable processable) {
            if (allocaOp)
              return allocaOp;
            if (auto found = allocaMap.find(primaryOp);
                found != allocaMap.end())
              return found->second;

            // Build a new allocation to store the intermediate value.
            OpBuilder allocBuilder = OpBuilder::atBlockBegin(&func.front());
            Location erasedLoc = UnknownLoc::get(primaryOp->getContext());
            auto allocSize = allocBuilder.create<LLVM::ConstantOp>(
                erasedLoc, allocBuilder.getI32Type(), 1);

            LLVM::AllocaOp newAllocaOp = allocBuilder.create<LLVM::AllocaOp>(
                erasedLoc, LLVM::LLVMPointerType::get(targetValue.getContext()),
                targetValue.getType(), allocSize, 0);
            if (anyMutable)
              allocaMap.insert({primaryOp, newAllocaOp});
            else
              allocaOp = newAllocaOp;
            return newAllocaOp;
          };

      llvm::SetVector<Operation *> opsToErase;
      llvm::MapVector<LLVM::DbgValueOp, Value> oldValueMap;

      // Converter for a single DbgValueOp.
      // `hasLimitedScope` controls whether the DbgValueOp is converted into a
      // DbgDeclareOp or is kept, because the scope of a DbgDeclareOp cannot be
      // limited to a subset of the parent scope.
      auto convertDbgValue =
          [&, targetValue = primaryValue](
              LLVM::DbgValueOp op, bool hasLimitedScope,
              DebugVariableSummary::ProcessableVariable &processable) {
            if (hasLimitedScope) {
              ArrayRef<LLVM::DIExpressionElemAttr> location =
                  op.getLocationExpr().getOperations();
              SmallVector<LLVM::DIExpressionElemAttr> newLocations = {
                  LLVM::DIExpressionElemAttr::get(
                      op.getContext(), llvm::dwarf::DW_OP_deref, {})};
              newLocations.append(location.begin(), location.end());
              oldValueMap.insert({op, op.getValue()});
              op.setOperand(getAllocaOp(processable));
              op.setLocationExprAttr(
                  LLVM::DIExpressionAttr::get(op->getContext(), newLocations));
            } else {
              ArrayRef<LLVM::DIExpressionElemAttr> location =
                  op.getLocationExpr().getOperations();
              if (!isa<BlockArgument>(op.getValue()) && !anyMutable &&
                  !location.empty() &&
                  location.front().getOpcode() == llvm::dwarf::DW_OP_deref &&
                  isa<LLVM::LLVMPointerType>(targetValue.getType())) {
                // For cases where the locationExpr begins with a deref, just
                // pop off the initial deref and convert directly into a
                // DbgDeclareOp. In this case no alloca needs to be created.
                // Block args however are not compatible directly with
                // DbgDeclare.
                auto refLocation = LLVM::DIExpressionAttr::get(
                    op->getContext(), location.drop_front());
                OpBuilder(op).create<LLVM::DbgDeclareOp>(
                    op.getLoc(), targetValue, op.getVarInfo(), refLocation);
              } else {
                // For all other cases, create alloca and use dbg declare with
                // it.
                OpBuilder(op).create<LLVM::DbgDeclareOp>(
                    op.getLoc(), getAllocaOp(processable), op.getVarInfo(),
                    op.getLocationExpr());
              }
              for (LLVM::DbgValueOp valueOp : processable.valueOps)
                opsToErase.insert(valueOp);
              for (LLVM::DbgValueOp valueOp : processable.additionalUndefs)
                opsToErase.insert(valueOp);
              if (processable.primaryUndef.has_value())
                opsToErase.insert(processable.primaryUndef.value());
            }
          };

      // Run converter on all single-def variables.
      bool hasUndefs = !processable.additionalUndefs.empty();

      // If additional undef dbg.values exist for this variable, cannot create
      // a dbg.declare for it as they don't allow undef dbg.values to limit
      // their live ranges. Keep the dbg.value ops but reference the stack
      // allocation instead.
      // Otherwise, if it only has one dbg.value, replace the old dbg.value
      // with a dbg.declare.
      convertDbgValue(primaryOp, hasUndefs, processable);

      // If no alloca was created for this value, nothing else is needed.
      // Otherwise, all users of the original value need to go thru the alloca.
      if (!allocaOp && allocaMap.size() == 0)
        continue;

      auto updateUses =
          [&](LLVM::DbgValueOp valueOp,
              DebugVariableSummary::ProcessableVariable processable) {
            // Update all of the old value uses to route through the alloca
            // instead of using the value directly.

            Value oldValue = valueOp.getValue();
            if (oldValueMap.contains(valueOp))
              oldValue = oldValueMap.lookup(valueOp);

            for (auto it = oldValue.user_begin(), e = oldValue.user_end();
                 it != e;) {
              // Grab the next unique user.
              Operation *user = *it;
              while (++it != e && *it == user)
                continue;

              if (opsToErase.contains(user))
                continue;

              // If the user is another dbg.value, it must be for a variable
              // that has multiple non-undef definitions. Cannot convert to
              // dbg.declare as it has a limited scope.
              if (auto dbgUser = dyn_cast<LLVM::DbgValueOp>(user)) {
                if (dbgUser.getValue().getDefiningOp<LLVM::UndefOp>()) {
                  convertDbgValue(dbgUser, true, processable);
                }
                continue;
              }
            }

            // Store into the alloca at the place where the value was defined.
            if (auto *valueOp = oldValue.getDefiningOp()) {
              OpBuilder storeBuilder(valueOp->getNextNode());
              storeBuilder.create<LLVM::StoreOp>(oldValue.getLoc(), oldValue,
                                                 getAllocaOp(processable));
            } else {
              // If the value is a block argument, we need to search for an
              // insertion point after the start of the block.
              auto insertPt = oldValue.getParentBlock()->begin();
              while (isa<LLVM::DbgValueOp, LLVM::DbgDeclareOp, LLVM::AllocaOp,
                         LLVM::ConstantOp>(*insertPt))
                ++insertPt;

              // Block arguments might not contain debuginfo scope (which can
              // trip up verifiers later), so to keep it simple, we also use
              // erasedLoc.
              OpBuilder storeBuilder(&*insertPt);
              Location erasedLoc = UnknownLoc::get(oldValue.getContext());
              storeBuilder.create<LLVM::StoreOp>(erasedLoc, oldValue,
                                                 getAllocaOp(processable));
            }
          };

      if (processable.valueOps.empty())
        updateUses(processable.primaryUndef.value(), processable);
      for (LLVM::DbgValueOp valueOp : processable.valueOps)
        updateUses(valueOp, processable);

      for (Operation *op : opsToErase) {
        op->erase();
      }
    }
  }
}
