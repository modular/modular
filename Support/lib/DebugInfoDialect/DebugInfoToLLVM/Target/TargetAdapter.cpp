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
    bool anyDerefs = false;
    /// Whether the deref paths are the same length (IE same number of pointer
    /// dereferences to get to the variable).
    bool allDerefsMatchedLength = true;
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
      auto [iter, inserted] = debugValuesToProcess.try_emplace(
          op.getVarInfo(), op, /*isUndef=*/true);
      if (!inserted)
        iter->second.additionalUndefs.push_back(op);
      return;
    }

    // Not undef.
    auto [iter, inserted] = debugValuesToProcess.try_emplace(
        op.getVarInfo(), op, /*isUndef=*/false);
    if (!inserted)
      iter->second.valueOps.push_back(op);
  });

  for (auto &[var, processable] : debugValuesToProcess) {
    // Check that the locationExpr is well formed.  We can handle locationExprs
    // that are 0 or more DW_OP_deref followed by zero or more
    // DW_OP_LLVM_fragment, but no fragments before deref, and no other
    // operations.
    bool onFirstPass = true;
    uint64_t firstNumDerefs = 0;
    for (auto valueOp : processable.valueOps) {
      if (processable.skip)
        break;
      bool foundFragment = false;
      uint64_t numDerefs = 0;
      for (auto exprOp : valueOp.getLocationExpr().getOperations()) {
        if (exprOp.getOpcode() == llvm::dwarf::DW_OP_deref) {
          processable.anyDerefs = true;
          ++numDerefs;
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
        firstNumDerefs = numDerefs;
      else if (firstNumDerefs != numDerefs)
        processable.allDerefsMatchedLength = false;
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
  for (auto func : module.getOps<mlir::FunctionOpInterface>()) {
    DebugVariableSummary debugVariableSummary =
        filterAndSummarizeDebugVariables(func);

    for (auto &mapElem : debugVariableSummary.processableVariableMap) {
      LLVM::DILocalVariableAttr &varInfo = mapElem.first;
      DebugVariableSummary::ProcessableVariable &processable = mapElem.second;

      // Don't build debug information for simple constants.
      if (processable.valueOps.size() == 1 &&
          processable.valueOps[0]
              .getValue()
              .getDefiningOp<LLVM::ConstantOp>() &&
          isa<IntegerType, FloatType>(
              processable.valueOps[0].getValue().getType()))
        continue;

      // For the useDerefMode case, we switch from dbg.value to dbg.declare,
      // but we only allocate if the SSA value for the dbg.value is a block
      // argument.
      bool useDerefMode =
          processable.anyDerefs && processable.allDerefsMatchedLength &&
          !processable.anyFragments && processable.additionalUndefs.empty();
      // For the declareDirectMode case, we switch from dbg.value to
      // dbg.declare, and we assemble struct fragments where applicable.  But it
      // requires that we have SSA values with no pointers in the exprLocation.
      bool declareDirectMode =
          !processable.anyDerefs && processable.additionalUndefs.empty();
      // For everything else, there's dbg.value.  If there are fragments behind
      // pointers, we would have to load them to assemble them in one place to
      // use dbg.declare, which is unsafe since some of the pointers may be null
      // or otherwise invalid.  If there are inconsistent levels of indirection,
      // we would have to do a load on the longer path to arrive at the same
      // pointer path length, which is unsafe.  Finally if there are undef
      // values (after the first non-undef value), it means that the variable
      // does not exist for the full scope of the function, and dbg.declare
      // implies a lifetime for the whole scope.  Note that where values are in
      // SSA values, we still make stack allocations for this case to ensure
      // that variables are in memory for GPU debugging.
      bool useDbgValueMode = !(useDerefMode || declareDirectMode);

      auto valueOpHasDeref = [](LLVM::DbgValueOp valueOp) -> bool {
        if (valueOp.getLocationExpr().getOperations().empty())
          return false;
        return valueOp.getLocationExpr().getOperations()[0].getOpcode() ==
               llvm::dwarf::DW_OP_deref;
      };

      // The converted alloca op that will hold the value if it is converted
      // to a stack allocation.  In the useDbgValue case, we need a separate
      // alloca for each value that is not already in memory.
      LLVM::AllocaOp allocaOp;
      llvm::MapVector<LLVM::DbgValueOp, LLVM::AllocaOp> allocaMap;

      // Get the allocaOp for this value. If one has not already been created,
      // create one and save it for the next invocation.
      auto getAllocaOp = [&](LLVM::DbgValueOp valueOp,
                             bool create = true) -> LLVM::AllocaOp {
        if (!useDbgValueMode && allocaOp)
          return allocaOp;
        if (allocaMap.contains(valueOp))
          return allocaMap.lookup(valueOp);
        if (!create)
          return {};

        // Build a new allocation to store the intermediate value.
        OpBuilder allocBuilder = OpBuilder::atBlockBegin(&func.front());
        Location erasedLoc = UnknownLoc::get(varInfo.getContext());

        int allocElems = 1;
        Type allocType = valueOp.getValue().getType();

        if (declareDirectMode && processable.anyFragments) {
          uint64_t sizeInBits = 0;
          if (auto t = dyn_cast<LLVM::DIBasicTypeAttr>(varInfo.getType()))
            sizeInBits = t.getSizeInBits();
          else if (auto t =
                       dyn_cast<LLVM::DICompositeTypeAttr>(varInfo.getType()))
            sizeInBits = t.getSizeInBits();
          else if (auto t =
                       dyn_cast<LLVM::DIDerivedTypeAttr>(varInfo.getType()))
            sizeInBits = t.getSizeInBits();
          allocElems = (sizeInBits / 8) + (sizeInBits % 8 ? 1 : 0);
          allocType = allocBuilder.getI8Type();
        }
        auto allocSize = allocBuilder.create<LLVM::ConstantOp>(
            erasedLoc, allocBuilder.getI32Type(), allocElems);

        // TODO - this just uses the default (0) address space, even if the
        // variables we are debugging actually live in a different address
        // space.
        Type pointerType = LLVM::LLVMPointerType::get(varInfo.getContext());

        LLVM::AllocaOp newAlloc = allocBuilder.create<LLVM::AllocaOp>(
            erasedLoc, pointerType, allocType, allocSize, 0);
        if (!useDbgValueMode)
          allocaOp = newAlloc;
        else
          allocaMap.insert({valueOp, newAlloc});
        return newAlloc;
      };

      bool eraseValueOps = false;
      llvm::DenseMap<LLVM::DbgValueOp, Value> oldValueMap;

      // Convert all processable variables.
      for (LLVM::DbgValueOp valueOp : processable.valueOps) {
        if (useDbgValueMode) {
          // If we are keeping dbg.value ops and it is already a pointer, there
          // is nothing to do.
          if (valueOpHasDeref(valueOp))
            continue;
          ArrayRef<LLVM::DIExpressionElemAttr> location =
              valueOp.getLocationExpr().getOperations();
          SmallVector<LLVM::DIExpressionElemAttr> newLocations = {
              LLVM::DIExpressionElemAttr::get(valueOp.getContext(),
                                              llvm::dwarf::DW_OP_deref, {})};
          newLocations.append(location.begin(), location.end());
          oldValueMap.insert({valueOp, valueOp.getValue()});
          valueOp.setOperand(getAllocaOp(valueOp));
          valueOp.setLocationExprAttr(
              LLVM::DIExpressionAttr::get(valueOp->getContext(), newLocations));
        } else {
          // Use dbg.declare.
          ArrayRef<LLVM::DIExpressionElemAttr> locationOps =
              valueOp.getLocationExpr().getOperations();
          Value declareOpArg = valueOp.getValue();
          if (useDerefMode) {
            if (isa<BlockArgument>(valueOp.getValue())) {
              // Block args are not compatible directly with DbgDeclare.  So we
              // allocate to add indirection to make it compatible.
              declareOpArg = getAllocaOp(valueOp);
            } else {
              // We need one less deref when switching to dbg.declare if we
              // aren't allocating and adding indirection.
              locationOps = locationOps.drop_front();
            }
          } else {
            // declareDirectMode case
            declareOpArg = getAllocaOp(valueOp);
            // The declareDirectMode case has an empty expression path, it is
            // simply the whole variable.
            locationOps = SmallVector<LLVM::DIExpressionElemAttr>();
          }
          OpBuilder(valueOp).create<LLVM::DbgDeclareOp>(
              valueOp.getLoc(), declareOpArg, valueOp.getVarInfo(),
              LLVM::DIExpressionAttr::get(valueOp->getContext(), locationOps));
          eraseValueOps = true;
        }
      }

      // Store values to the allocations
      for (LLVM::DbgValueOp valueOp : processable.valueOps) {
        if (!getAllocaOp(valueOp, /*create=*/false))
          continue;

        Value oldValue = valueOp.getValue();
        if (useDbgValueMode)
          oldValue = oldValueMap.lookup(valueOp);
        Location erasedLoc = UnknownLoc::get(oldValue.getContext());
        Type pointerType = LLVM::LLVMPointerType::get(oldValue.getContext());

        // We are only dealing with fragments in cases where there are no
        // pointers.
        uint64_t offsetInBits = 0;
        // We only want to actually store at an offset if we are using
        // dbg.declare to hold the full struct, and we need to store
        // individual fields in it.
        if (declareDirectMode) {
          for (auto exprOp : valueOp.getLocationExpr().getOperations()) {
            if (exprOp.getOpcode() == llvm::dwarf::DW_OP_LLVM_fragment) {
              assert(exprOp.getArguments().size() == 2 &&
                     "bad DW_OP_LLVM_fragment");
              offsetInBits += exprOp.getArguments()[0];
            }
          }
        }

        auto makeStore = [&](OpBuilder storeBuilder, Location storeLoc) {
          Value storeToPointer = getAllocaOp(valueOp);
          if (offsetInBits != 0) {
            uint64_t offsetInBytes = offsetInBits / 8;
            assert(offsetInBits % 8 == 0 && "offset makes sense");
            LLVM::GEPOp uglyGep = storeBuilder.create<LLVM::GEPOp>(
                erasedLoc, /*resultType=*/pointerType,
                /*elementType=*/storeBuilder.getI8Type(),
                /*basePtr=*/storeToPointer, LLVM::GEPArg(offsetInBytes));
            storeToPointer = uglyGep;
          }
          storeBuilder.create<LLVM::StoreOp>(storeLoc, oldValue,
                                             storeToPointer);
        };

        // Store into the alloca at the place where the value was defined.
        if (auto *definingOp = oldValue.getDefiningOp()) {
          makeStore(OpBuilder(definingOp->getNextNode()), oldValue.getLoc());
        } else {
          // If the value is a block argument, we need to search for an
          // insertion point after the start of the block.
          auto insertPt = oldValue.getParentBlock()->begin();
          while (isa<LLVM::DbgValueOp, LLVM::DbgDeclareOp, LLVM::AllocaOp,
                     LLVM::ConstantOp>(*insertPt)) {
            ++insertPt;
          }

          // Block arguments might not contain debuginfo scope (which can
          // trip up verifiers later), so to keep it simple, we also use
          // erasedLoc.
          OpBuilder storeBuilder(&*insertPt);
          makeStore(storeBuilder, erasedLoc);
        }
      }

      // If we switched away from dbg.value to dbg.declare, we need to erase the
      // old ops.
      if (eraseValueOps) {
        if (processable.primaryUndef.has_value())
          processable.primaryUndef.value()->erase();
        for (Operation *op : processable.additionalUndefs)
          op->erase();
        for (Operation *op : processable.valueOps)
          op->erase();
      }
    }
  }
}
