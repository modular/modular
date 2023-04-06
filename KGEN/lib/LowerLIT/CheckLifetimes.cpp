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

/// Find all the functions and types in the module.
static std::pair<std::vector<LIT::FuncOp>,
                 DenseMap<SymbolRefAttr, LIT::StructDeclOp>>
collectFunctionsAndTypes(Operation *module) {
  std::vector<LIT::FuncOp> funcList;
  DenseMap<SymbolRefAttr, LIT::StructDeclOp> structMap;
  module->walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) -> mlir::WalkResult {
        if (auto funcOp = dyn_cast<LIT::FuncOp>(op)) {
          funcList.push_back(funcOp);
          // Skip recursing into the function, all nested functions will be
          // handled separately, and structs don't currently live in functions.
          return WalkResult::skip();
        }
        if (auto structOp = dyn_cast<LIT::StructDeclOp>(op))
          structMap[getFullyResolvedSymbolRef(structOp)] = structOp;
        return WalkResult::advance();
      });
  return {std::move(funcList), std::move(structMap)};
}

//===----------------------------------------------------------------------===//
// TypeDeclInfo
//===----------------------------------------------------------------------===//

/// Information about a struct declarations, used for field sensitive analysis.
/// Value tracking is completely field sensitive, tracking values at the level
/// of individual fields in their flattened representation.  To do this, we need
/// an efficient mapping that tells us the number of (fully flattened) fields in
/// struct.
struct TypeDeclInfo {
  TypeDeclInfo(DenseMap<SymbolRefAttr, LIT::StructDeclOp> &&structMap)
      : structMap(std::move(structMap)) {}

  /// Return the total number of flattened fields in the specified type.
  unsigned getNumFieldsInType(Type type);

  /// Return the start bit for a field with the specified name in the specified
  /// type.
  unsigned getFieldIndex(DeclRefType type, StringAttr fieldName);

private:
  DenseMap<SymbolRefAttr, LIT::StructDeclOp> structMap;

  /// This keeps track of the number of fields in the struct specified by the
  /// (fully flattened) symbol.
  DenseMap<SymbolRefAttr, unsigned> numFields;

  /// A map from struct name and field name to index within the struct.  This
  /// isn't the field number, this is the number of recursively flattened
  /// fields until the start of the field.
  DenseMap<std::pair<SymbolRefAttr, StringAttr>, unsigned> fieldIndices;
};

/// Return the total number of flattened fields in the specified type.
unsigned TypeDeclInfo::getNumFieldsInType(Type type) {
  // We currently treat all non-struct types as being a single element, even
  // things like kgen.list containing struct types.
  DeclRefType declRef = dyn_cast<DeclRefType>(type);
  if (!declRef)
    return 1;

  // See if we've already looked this up, if so, just return the known value.
  SymbolRefAttr structSymbol = declRef.getSymbol();
  auto it = numFields.find(structSymbol);
  if (it != numFields.end())
    return it->second;

  // If not, we compute it recursively.  Structs cannot be infinitely deep, so
  // we can just do this recursively.
  LIT::StructDeclOp decl = structMap[structSymbol];
  assert(decl && "reference to struct that wasn't declared");

  size_t totalFields = 0;
  for (auto field : decl.getFieldDecls()) {
    fieldIndices[{structSymbol, field.getNameAttr()}] = totalFields;
    totalFields += getNumFieldsInType(field.getType());
  }
  return numFields[structSymbol] = totalFields;
}

/// Return the start bit for a field with the specified name in the specified
/// type.
unsigned TypeDeclInfo::getFieldIndex(DeclRefType type, StringAttr fieldName) {
  return fieldIndices[{type.getSymbol(), fieldName}];
}

//===----------------------------------------------------------------------===//
// ValueInfo / ValueSet tracking
//===----------------------------------------------------------------------===//

namespace {
struct ValueInfo {
  /// This is the declared value being tracked.
  const Value value;

  /// This is the destructor for the value if it exists, otherwise may be null.
  const TypedAttr dtor;

  /// This indicates the (first, end] bitrange in the bit vector corresponding
  /// to this value.
  const unsigned startValueBit, endValueBit;

  /// True if this values starts out uninitialized at the beginning of its
  /// lifetime.
  const bool startsUninit;
  /// True if this value needs to be uninitialized at the end of its lifetime.
  const bool endsUninit;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;
};

/// A ValueRef indicates a slice reference into the bitvector for all the
/// values.
struct ValueRef {
  /// This is the entry # for the ValueInfo for the overall value.
  unsigned valueId;

  /// This is the (start, end] span of bits for the reference that we're
  /// tracking, which may be a subset of the overall value.
  unsigned startBit, endBit;

  ValueRef() : valueId(0), startBit(~0U), endBit(~0U) {}
  ValueRef(unsigned valueId, unsigned startBit, unsigned endBit)
      : valueId(valueId), startBit(startBit), endBit(endBit) {}

  /// Allow use of a ValueRef in a boolean condition.
  operator bool() const { return valueId != 0; }

  /// Test if all the bits in the range are set in the specified bitvector.
  bool isAllPresent(const BitVector &bits) const {
    // BitVector doesn't have a more efficient method for this.  We could make
    // this more efficient for longer ranges if needed.
    for (size_t i = startBit, e = endBit; i != e; ++i)
      if (!bits[i])
        return false;
    return true;
  }

  /// Set the bits for this range to zero or one in the specified bitvector.
  void markBits(BitVector &bits, bool newValue) const {
    if (!valueId)
      return;
    if (newValue)
      bits.set(startBit, endBit);
    else
      bits.reset(startBit, endBit);
  }
};

/// This tracks the values in a function (including nested functions) that are
/// relevant for ownership - that needs to be tracked for uses without being
/// initialized, or that need a destructor to be run.
///
/// This tracks a /completely field sensitive/ view of the values under
/// consideration, including their nested fields in a flattened representation.
/// This gives us a fully precise view of the individual fields, and allows them
/// to be initialized and consumed in a piecewise way.
struct ValueSet {
  /// Initialize the value set with one entry, so index #0 is always invalid and
  /// can be used as a sentinel, and so a null Value is always treated as
  /// untracked.
  ValueSet(TypeDeclInfo &typeDeclInfo) : typeDeclInfo(typeDeclInfo) {
    addValue(Value(), false, TypedAttr(), false, false);
  }

  /// Return the number of values we are tracking.
  MutableArrayRef<ValueInfo> getValueInfos() { return valueInfos; }

  /// Add a value to the set that we are tracking.  This includes:
  ///  * the MLIR representation for the value itself
  ///  * whether the value is a by-ref pointer to the underlying logical value
  ///  * the destructor for the value
  ///  * whether the value starts out uninit or init at the function start
  ///  * whether the value is uninit or init at normal function return.
  ///
  void addValue(Value val, bool isByRef, TypedAttr dtor, bool startsUninit,
                bool endsUninit) {
    // FIXME(staging): Ignore values without destructors.
    if (!ENABLE_FOR_ALL && !dtor && val)
      return;

    unsigned firstValueBit = getNumTotalBits();
    unsigned numValueBits = 1;
    // We are only field sensitive for memory objects, not in-register values.
    // It isn't possible to update in-register values: register_passable values
    // are always valid when present.  If they pass through memory, they are
    // checked when loaded from memory.
    if (val && isByRef) {
      Type valType =
          cast<POP::PointerType>(val.getType()).getResolvedElementType();
      numValueBits = typeDeclInfo.getNumFieldsInType(valType);
    }

    valueInfoIndex[val] = valueInfos.size();
    valueInfos.push_back({val, dtor, firstValueBit,
                          firstValueBit + numValueBits, startsUninit,
                          endsUninit, /*hasErrorDiagnosed=*/false});
  }

  /// If this value is directly tracked by the ValueSet, return the index of the
  /// value, otherwise return zero.
  ValueRef getDirectValueRef(Value value) const {
    auto it = valueInfoIndex.find(value);
    if (it == valueInfoIndex.end())
      return ValueRef();
    auto &entry = valueInfos[it->second];
    return ValueRef{it->second, entry.startValueBit, entry.endValueBit};
  }

  /// Given a pointer that is being accessed indirectly by an operation, return
  /// the value number being referenced, or zero if not tracked.
  ValueRef getPointerValueIndex(Value value);

  /// Return the total number of bits we need to track in the bitvector.
  unsigned getNumTotalBits() const {
    return !valueInfos.empty() ? valueInfos.back().endValueBit : 0;
  }

private:
  TypeDeclInfo &typeDeclInfo;
  SmallVector<ValueInfo> valueInfos;
  DenseMap<Value, unsigned> valueInfoIndex;
};
} // namespace

/// Given a pointer that is being accessed indirectly by an operation, return
/// the value number being referenced, or zero if not tracked.
ValueRef ValueSet::getPointerValueIndex(Value value) {
  // If this is a GEP, check the base and focus in on a field of it.
  if (auto structGEP = value.getDefiningOp<StructGEPOp>()) {
    ValueRef baseVal = getPointerValueIndex(structGEP.getContainer());
    if (!baseVal)
      return baseVal;

    // Figure out what subset of elements we have indexed to.
    auto containerType =
        structGEP.getContainer().getType().getResolvedElementType();
    unsigned fieldOffset = typeDeclInfo.getFieldIndex(
        cast<DeclRefType>(containerType), structGEP.getFieldAttr());
    unsigned startBit = baseVal.startBit + fieldOffset;
    auto resultType = structGEP.getType().getResolvedElementType();
    return ValueRef{baseVal.valueId, startBit,
                    startBit + typeDeclInfo.getNumFieldsInType(resultType)};
  }

  // Otherwise, we don't know what this is.
  return getDirectValueRef(value);
}

//===----------------------------------------------------------------------===//
// UninitializedValueScan
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the second pass over a function body, which
/// identifies and complains about uses of uninitialized values.
struct UninitializedValueScan {
  UninitializedValueScan(ValueSet &valueSet) : valueSet(valueSet) {}
  UninitializedValueScan(const UninitializedValueScan &existing) = delete;

  void scanFunction(LIT::FuncOp func);
  void scanBlock(Block &body);

private:
  void checkOp(Operation &op);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is the set of values known to be live at this point.
  BitVector liveValues;

  /// When analyzing the body of a loop, this bitset indicates what a 'continue'
  /// should intersect with.
  BitVector *continueSet = nullptr;
  /// When analyzing the body of a loop, this bitset indicates what a 'break'
  /// should intersect with.
  BitVector *breakSet = nullptr;
  /// When analyzing the body of a try, this bitset indicates what a 'raise'
  /// should intersect with.
  BitVector *raiseSet = nullptr;
};
} // namespace

void UninitializedValueScan::scanFunction(LIT::FuncOp func) {
  // Initialize the BitVector with all the elements that are live-in.  We treat
  // all values live at the start of the function (even before they are actually
  // defined) because we know that all uses must be after them due to SSA
  // dominance.
  liveValues.resize(valueSet.getNumTotalBits());
  for (const ValueInfo &valueInfo : valueSet.getValueInfos())
    if (!valueInfo.startsUninit)
      liveValues.set(valueInfo.startValueBit, valueInfo.endValueBit);

  // Scan the body of the function.
  scanBlock(*func.getBody());
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void UninitializedValueScan::scanBlock(Block &block) {
  for (Operation &op : block)
    checkOp(op);
}

void UninitializedValueScan::checkOp(Operation &op) {
  // This op is handled when used.
  if (isa<StructGEPOp>(op))
    return;

  // Verify that the specified value ID is live at this point, diagnosing an
  // error if not.
  auto checkLive = [&](ValueRef valueRef) {
    if (valueRef.isAllPresent(liveValues))
      return;
    ValueInfo &valueEntry = valueSet.getValueInfos()[valueRef.valueId];
    if (!valueEntry.hasErrorDiagnosed) {
      auto diag =
          mlir::emitError(op.getLoc(), "invalid use of uninitialized value");
      diag.attachNote(valueEntry.value.getLoc()) << "value declared here";
      valueEntry.hasErrorDiagnosed = true;
    }
  };

  auto checkSSAValueLive = [&](Value value) -> ValueRef {
    ValueRef valueId = valueSet.getDirectValueRef(value);
    if (valueId)
      checkLive(valueId);
    return valueId;
  };

  auto checkDirectPointerLive = [&](Value value) -> ValueRef {
    ValueRef valueRef = valueSet.getPointerValueIndex(value);
    if (valueRef)
      checkLive(valueRef);
    return valueRef;
  };

  // A store of a whole value is an initialization.
  if (auto storeOp = dyn_cast<POP::StoreOp>(op)) {
    // This marks its value live.
    valueSet.getPointerValueIndex(storeOp.getPtr()).markBits(liveValues, true);
    return;
  }

  // A load is a use of whatever fields are being referenced.
  // an initialization.
  if (auto loadOp = dyn_cast<POP::LoadOp>(op)) {
    // This marks its value live.
    checkLive(valueSet.getPointerValueIndex(loadOp.getPtr()));
    return;
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
        checkSSAValueLive(operand).markBits(liveValues, false);
        break;
      case ValueInputConvention::BorrowedInReg:
        // Live -> live.
        checkSSAValueLive(operand);
        break;
      case ValueInputConvention::OwnedInMem:
        // Transitions live -> dead.
        checkDirectPointerLive(operand).markBits(liveValues, false);
        break;
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::ByRef:
        // Live -> live.
        checkDirectPointerLive(operand);
        break;
      case ValueInputConvention::ByRefResult:
        // This call defines the by-ref result.
        valueSet.getPointerValueIndex(operand).markBits(liveValues, true);
        break;
      }
    }
    return;
  }

  // If this operation has a direct use of a value we are tracking, consider
  // it a use that must be initialized.  This notably includes LoadOp.
  bool hasUse = false;
  for (Value operand : op.getOperands())
    hasUse |= checkSSAValueLive(operand) != 0;

  // If this is a kgen.return then we have an exit from the function
  // (including early returns and exception raises that leave the function).
  // Check that all of the values we are tracking are managed correctly.
  if (isa<KGEN::ReturnOp>(op)) {
    auto valueInfosRef = valueSet.getValueInfos();
    for (size_t i = 1, e = valueInfosRef.size(); i != e; ++i)
      if (!valueInfosRef[i].endsUninit)
        checkLive(ValueRef{unsigned(i), valueInfosRef[i].startValueBit,
                           valueInfosRef[i].endValueBit});
    return;
  }

  // An unreachable at the end of the block considers all values live, which
  // makes it flexible when merging with any other control flow.
  if (isa<KGEN::UnreachableOp>(op)) {
    liveValues.set();
    return;
  }

  if (isa<HLCF::BreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    *breakSet &= liveValues;
    return;
  }
  if (isa<HLCF::ContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    *continueSet &= liveValues;
    return;
  }
  if (isa<LIT::TryRaiseOp>(op)) {
    assert(raiseSet && "Not in a 'try'?");
    *raiseSet &= liveValues;
    return;
  }

  // 'if' operations treat the condition as a use but have live outs that are
  // the intersection of the live values produced by the then/else branches.
  if (isa<HLCF::IfOp, ParamIfOp>(op)) {
    assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
           op.getRegion(1).hasOneBlock() &&
           "if-like op should have two single-block regions");
    BitVector liveValuesCopy = liveValues;
    scanBlock(op.getRegion(0).front());
    liveValuesCopy.swap(liveValues);
    scanBlock(op.getRegion(1).front());
    liveValues &= liveValuesCopy;
    return;
  }

  // For a loop, we analyze the body of the loop with the known live-ins but
  // capture a new sets for continue and break results.
  if (auto loopOp = dyn_cast<HLCF::LoopOp>(op)) {
    UninitializedValueScan bodySets(valueSet);
    // Loops are transparent to raise.
    bodySets.raiseSet = raiseSet;

    // The default continueSet is the live-in set of values.  This can lose
    // values if some 'continue' path through the body of the loop consumes a
    // value.
    BitVector continueSet(liveValues);
    bodySets.continueSet = &continueSet;

    // The 'breakSet' of the loop body will be the live outs of the loop.  We
    // use the existing liveValues that we'll continue with for that set, but
    // need to start it out thinking that everything is live so intersections
    // from the body work correctly.
    liveValues.set();
    bodySets.breakSet = &liveValues;

    // Iteratively scan the loop body until the live-in set converges.  This is
    // a trivial lattice with each bit converging to "not live in", so we know
    // this will terminate.
    size_t numLiveIn = continueSet.count();
    while (1) {
      // Scan the body: any breaks will intersect their live-out set with
      // 'breakSet', and any continues will intersect their live-out set with
      // 'continueSet'.
      bodySets.liveValues = continueSet;
      bodySets.scanBlock(loopOp.getBody().front());

      // If any bits got cleared from the continueSet then we need to iterate.
      size_t newLiveIn = continueSet.count();
      if (newLiveIn == numLiveIn)
        break;
      numLiveIn = newLiveIn;
    }
    // Any code after the loop continues on with the breaks valid.
    return;
  }

  if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
    UninitializedValueScan bodySets(valueSet);
    // Our current live-in set is live-in to the try body.
    bodySets.liveValues = liveValues;

    // Try is transparent to break/continue.
    bodySets.continueSet = continueSet;
    bodySets.breakSet = breakSet;

    // We capture all the common values live-out of raise's as being the live-in
    // to the except block.
    BitVector exceptSet(liveValues.size(), true);
    bodySets.raiseSet = &exceptSet;
    bodySets.scanBlock(tryOp.getTryRegion().front());

    // The live-ins to the except block are the exceptSet.
    liveValues = std::move(exceptSet);
    scanBlock(tryOp.getExceptRegion().front());

    // The live-out set of the bodySet is the live-in to the else block, but
    // exceptions raised in it go out of the try.
    bodySets.raiseSet = raiseSet;
    bodySets.scanBlock(tryOp.getElseRegion().front());

    // The fall through live values are the intersection from the except and
    // else blocks.
    liveValues &= bodySets.liveValues;
    return;
  }

  (void)hasUse;
#if STAGING
    if (hasUse && !isMemoryEffectFree(&op) && !isa<POP::LoadOp>(op))
      op.dump();
#endif
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
    // Find all the functions and structs in the module.
    auto [functionVector, structMap] = collectFunctionsAndTypes(getOperation());

    // Process all the structs into TypeDeclInfo.
    TypeDeclInfo typeDeclInfo(std::move(structMap));

    // TODO: Do in parallel, watch out for mutations of TypeDeclInfo though!
    bool hadError = false;
    for (auto func : functionVector)
      hadError |= failed(processFunction(func, typeDeclInfo));

    if (hadError)
      return signalPassFailure();
  }

  LogicalResult processFunction(LIT::FuncOp func, TypeDeclInfo &typeDeclInfo);
};
} // namespace

LogicalResult CheckLifetimes::processFunction(LIT::FuncOp func,
                                              TypeDeclInfo &typeDeclInfo) {
  // Pass #1: Collect all of the values declared in the function that have
  // ownership to track, and number them.
  SmallVector<OwnedArgDeclOp> ownedArgDecls;
  ValueSet valueSet(typeDeclInfo);
  auto collectArguments = [&](SignatureType signature, Block *body) {
    for (auto [convention, bbArg] : llvm::zip(
             signature.getValueInputConventions(), body->getArguments())) {
      if (convention == ValueInputConvention::ByRef ||
          convention == ValueInputConvention::ByRefResult)
        valueSet.addValue(bbArg, /*isByRef*/ true, /*no-dtor*/ TypedAttr(),
                          /*startsUninit*/ false, /*endsUninit*/ false);
    }
  };
  func->walk([&](Operation *op) {
    if (auto ownedArgDecl = dyn_cast<OwnedArgDeclOp>(op)) {
      ownedArgDecls.push_back(ownedArgDecl);
      return valueSet.addValue(op->getResult(0), /*isByRef*/ true,
                               ownedArgDecl.getDtorAttr(),
                               /*startsUninit*/ false, /*endsUninit*/ true);
    }
    // LetReg starts out initialized with its own value.
    if (auto letReg = dyn_cast<LetRegDeclOp>(op))
      return valueSet.addValue(op->getResult(0), /*isByRef*/ false,
                               letReg.getDtorAttr(),
                               /*startsUninit*/ false, /*endsUninit*/ true);
    // VarLetDeclOp is uninit and ends that way.
    if (auto varLet = dyn_cast<VarLetDeclOp>(op))
      return valueSet.addValue(op->getResult(0), /*isByRef*/ true,
                               varLet.getDtorAttr(),
                               /*startsUninit*/ true, /*endsUninit*/ true);

    // Collect values we need to track from arguments to closures and the outer
    // function itself.
    if (auto func = dyn_cast<LIT::FuncOp>(op))
      collectArguments(func.getSignature(), func.getBody());
    else if (auto func = dyn_cast<ParamDeclareRegionOp>(op))
      collectArguments(func.getSignature(), func.getBody());
  });

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined, emitting diagnostics as we go.
  UninitializedValueScan(valueSet).scanFunction(func);

  // TODO: How do we want to handle captures in closures?  Their uses
  // effectively form the capture list for the closure.  Should this get
  // materialized by LowerSemanticCF before this pass?

  // Finally, remove all the OwnedArgDeclOp's now that we're done with them.
  for (auto ownedArg : ownedArgDecls) {
    ownedArg.replaceAllUsesWith(ownedArg.getValue());
    ownedArg->erase();
  }

  // Return failure if we generated an error.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  }));
}
