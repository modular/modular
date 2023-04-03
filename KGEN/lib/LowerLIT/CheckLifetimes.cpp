//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass checks value lifetime invariants, e.g. that
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include <llvm/ADT/STLExtras.h>

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

/// FIXME: This flag enables the checker for all values, when this is clear, it
/// only enables it for things that have explicit destructors.  This helps stage
/// things in.
enum { ENABLE_FOR_ALL = 0 };

/// Find all the functions in the module, which may be buried in structures.
static std::vector<LIT::FuncOp> collectFunctions(Operation *op) {
  std::vector<LIT::FuncOp> result;
  op->walk<mlir::WalkOrder::PreOrder>(
      [&](LIT::FuncOp funcOp) -> mlir::WalkResult {
        result.push_back(funcOp);
        // Skip recursing into the function, all nested functions will be
        // handled separately.
        return WalkResult::skip();
      });
  return result;
}

namespace {
struct ValueInfo {
  /// This is the location of the value being declared.
  const Location loc;

  /// This is the destructor for the value if it exists, otherwise may be null.
  const TypedAttr dtor;

  /// True if this values starts out uninitialized at the beginning of its
  /// lifetime.
  const bool startsUninit;
  /// True if this value needs to be uninitialized at the end of its lifetime.
  const bool endsUninit;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;
};

/// This tracks the values in a function (including nested functions) that are
/// relevant for ownership - that needs to be tracked for uses without being
/// initialized, or that need a destructor to be run.
struct ValueSet {

  /// Initialize the value set with one entry, so index #0 is always invalid and
  /// can be used as a sentinel, and so a null Value is always treated as
  /// untracked.
  ValueSet(MLIRContext *context) : context(context) {
    addValue(Value(), TypedAttr(), false, false);
  }

  /// Return the number of values we are tracking.
  MutableArrayRef<ValueInfo> getValueInfos() { return valueInfos; }

  ValueInfo &operator[](size_t idx) { return valueInfos[idx]; }

  /// If this value is directly tracked by the ValueSet, return the index of the
  /// value, otherwise return zero.
  size_t getDirectValueIndex(Value value) const {
    auto it = memoryObjectIndex.find(value);
    return it != memoryObjectIndex.end() ? it->second : 0;
  }

  // Add a value to the set that we are tracking, along with whether the value
  // starts out initialized.
  void addValue(Value val, TypedAttr dtor, bool startsUninit, bool endsUninit) {
    // FIXME(staging): Ignore values without destructors.
    if (!ENABLE_FOR_ALL && !dtor && val)
      return;

    memoryObjectIndex[val] = valueInfos.size();
    Location loc = val ? val.getLoc() : mlir::UnknownLoc::get(context);
    valueInfos.push_back({loc, dtor, startsUninit, endsUninit, false});
  }

  /// Given a pointer that is being accessed indirectly by an operation, return
  /// the value number being referenced, or zero if not tracked.
  size_t getPointerValueIndex(Value value);

private:
  MLIRContext *const context;
  SmallVector<ValueInfo> valueInfos;
  DenseMap<Value, size_t> memoryObjectIndex;
};
} // namespace

/// Given a pointer that is being accessed indirectly by an operation, return
/// the value number being referenced, or zero if not tracked.
size_t ValueSet::getPointerValueIndex(Value value) {
  while (1) {
    // Check to see if this is directly involved.
    if (size_t index = getDirectValueIndex(value))
      return index;

    // If this is a GEP, check the base.
    if (auto structGEP = value.getDefiningOp<StructGEPOp>()) {
      value = structGEP.getContainer();
      continue;
    }

    // Otherwise, we don't know what this is.
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CHECKLIFETIMES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct CheckLifetimes : impl::CheckLifetimesBase<CheckLifetimes> {
  using CheckLifetimesBase::CheckLifetimesBase;

  void runOnOperation() override {
    // Find all the functions in the module.
    std::vector<LIT::FuncOp> functions = collectFunctions(getOperation());

    bool hadError = false;

    // TODO: Do in parallel.
    for (auto func : functions)
      hadError |= failed(processFunction(func));

    if (hadError)
      return signalPassFailure();
  }

  LogicalResult processFunction(LIT::FuncOp func);
  void scanForUninitializedValueUses(Block &block, BitVector &liveValues,
                                     ValueSet &valueSet);
};
} // namespace

LogicalResult CheckLifetimes::processFunction(LIT::FuncOp func) {
  // Pass #1: Collect all of the values declared in the function that have
  // ownership to track, and number them.
  SmallVector<OwnedArgDeclOp> ownedArgDecls;
  ValueSet valueSet(func.getContext());
  auto collectArguments = [&](SignatureType signature, Block *body) {
    for (auto [convention, bbArg] : llvm::zip(
             signature.getValueInputConventions(), body->getArguments())) {
      if (convention == ValueInputConvention::ByRef ||
          convention == ValueInputConvention::ByRefResult)
        valueSet.addValue(bbArg, /*no-dtor*/ TypedAttr(),
                          /*startsUninit*/ false, /*endsUninit*/ false);
    }
  };
  func->walk([&](Operation *op) {
    if (auto ownedArgDecl = dyn_cast<OwnedArgDeclOp>(op)) {
      ownedArgDecls.push_back(ownedArgDecl);
      return valueSet.addValue(op->getResult(0), ownedArgDecl.getDtorAttr(),
                               /*startsUninit*/ false, /*endsUninit*/ true);
    }
    // LetReg starts out initialized with its own value.
    if (auto letReg = dyn_cast<LetRegDeclOp>(op))
      return valueSet.addValue(op->getResult(0), letReg.getDtorAttr(),
                               /*startsUninit*/ false, /*endsUninit*/ true);
    // VarLetDeclOp is uninit and ends that way.
    if (auto varLet = dyn_cast<VarLetDeclOp>(op))
      return valueSet.addValue(op->getResult(0), varLet.getDtorAttr(),
                               /*startsUninit*/ true, /*endsUninit*/ true);

    // Collect values we need to track from arguments to closures and the outer
    // function itself.
    if (auto func = dyn_cast<LIT::FuncOp>(op))
      collectArguments(func.getSignature(), func.getBody());
    else if (auto func = dyn_cast<ParamDeclareRegionOp>(op))
      collectArguments(func.getSignature(), func.getBody());
  });

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined.

  // Initialize the BitVector with all the elements that are live-in.  We treat
  // all values live at the start of the function (even before they are actually
  // defined) because we know that all uses must be after them due to SSA
  // dominance.
  BitVector liveValues(valueSet.getValueInfos().size());
  for (auto [idx, valueInfo] : llvm::enumerate(valueSet.getValueInfos()))
    if (!valueInfo.startsUninit)
      liveValues.set(idx);

  scanForUninitializedValueUses(*func.getBody(), liveValues, valueSet);

  // TODO: How do we want to handle captures in closures?  Their uses
  // effectively form the capture list for the closure.  Should this get
  // materialized by LowerSemanticCF before this pass?

  // Finally, remove all the OwnedArgDeclOp's now that we're done with them.
  for (auto ownedArg : ownedArgDecls) {
    ownedArg.replaceAllUsesWith(ownedArg.getValue());
    ownedArg->erase();
  }

  // Return failure if we generated an error.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo info) {
    return info.hasErrorDiagnosed;
  }));
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void CheckLifetimes::scanForUninitializedValueUses(Block &block,
                                                   BitVector &liveValues,
                                                   ValueSet &valueSet) {
  for (Operation &op : block) {
    // Verify that the specified value ID is live at this point, diagnosing an
    // error if not.
    auto checkValueIdLive = [&](size_t valueId) {
      if (!liveValues[valueId] && !valueSet[valueId].hasErrorDiagnosed) {
        auto diag = op.emitError("invalid use of uninitialized value");
        diag.attachNote(valueSet[valueId].loc) << "value declared here";
        valueSet[valueId].hasErrorDiagnosed = true;
      }
    };

    auto checkSSAValueLive = [&](Value value) -> size_t {
      size_t valueId = valueSet.getPointerValueIndex(value);
      if (valueId)
        checkValueIdLive(valueId);
      return valueId;
    };

    auto checkDirectPointerLive = [&](Value value) -> size_t {
      size_t valueId = valueSet.getDirectValueIndex(value);
      if (valueId)
        checkValueIdLive(valueId);
      return valueId;
    };

    auto markValueIdState = [&](size_t valueId, bool isLive) {
      if (valueId)
        liveValues[valueId] = isLive;
    };

    // A store of a whole value is an initialization.
    if (auto storeOp = dyn_cast<POP::StoreOp>(op)) {
      // This marks its value live.
      markValueIdState(valueSet.getDirectValueIndex(storeOp.getPtr()), true);
      continue;
    }

    // If this is a call, investigate each of the operands along with the
    // argument convention effects.
    if (isa<KGEN::CallOp>(op)) { // TODO: Generalize
      auto call = dyn_cast<KGENCallOpInterface>(op);
      auto signature = call.getCalleeType();
      auto operands = call->getOperands();
      assert(signature.getValueInputConventions().size() == operands.size());
      for (auto [convention, operand] :
           llvm::zip(signature.getValueInputConventions(), operands)) {
        switch (convention) {
        case ValueInputConvention::OwnedInReg:
          // Transitions live -> dead.
          markValueIdState(checkSSAValueLive(operand), false);
          break;
        case ValueInputConvention::BorrowedInReg:
          // Live -> live.
          checkSSAValueLive(operand);
          break;
        case ValueInputConvention::OwnedInMem:
          // Transitions live -> dead.
          markValueIdState(checkDirectPointerLive(operand), false);
          break;
        case ValueInputConvention::BorrowedInMem:
        case ValueInputConvention::ByRef:
          // Live -> live.
          checkDirectPointerLive(operand);
          break;
        case ValueInputConvention::ByRefResult:
          // This call defines the by-ref result.
          markValueIdState(valueSet.getDirectValueIndex(operand), true);
          break;
        }
      }
      continue;
    }

    // If this operation has a direct use of a value we are tracking, consider
    // it a use that must be initialized.  This notably includes LoadOp.
    [[maybe_unused]] bool hasUse =
        llvm::any_of(op.getOperands(), [&](auto operand) {
          return checkSSAValueLive(operand) != 0;
        });

    // If this is a kgen.return then we have an exit from the function
    // (including early returns and exception raises that leave the function).
    // Check that all of the values we are tracking are managed correctly.
    if (isa<KGEN::ReturnOp>(op)) {
      auto valueInfosRef = valueSet.getValueInfos();
      for (size_t i = 1, e = valueInfosRef.size(); i != e; ++i)
        if (!valueInfosRef[i].endsUninit)
          checkValueIdLive(i);
      continue;
    }

    // An unreachable at the end of the block considers all values live, which
    // makes it flexible when merging with any other control flow.
    if (isa<KGEN::UnreachableOp>(op)) {
      liveValues.set();
      continue;
    }

    // 'if' operations treat the condition as a use but have live outs that are
    // the intersection of the live values produced by the then/else branches.
    if (isa<HLCF::IfOp, ParamIfOp>(op)) {
      assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
             op.getRegion(1).hasOneBlock() &&
             "if-like op should have two single-block regions");
      BitVector liveValuesCopy = liveValues;
      scanForUninitializedValueUses(op.getRegion(0).front(), liveValues,
                                    valueSet);
      scanForUninitializedValueUses(op.getRegion(1).front(), liveValuesCopy,
                                    valueSet);
      liveValues &= liveValuesCopy;
      continue;
    }

#if STAGING
    if (hasUse && !isMemoryEffectFree(&op) && !isa<POP::LoadOp>(op))
      op.dump();
#endif
  }
}
