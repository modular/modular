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

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

/// Find all the functions and types in the module.
static std::pair<std::vector<mlir::FunctionOpInterface>,
                 DenseMap<SymbolRefAttr, LIT::StructDeclOp>>
collectFunctionsAndTypes(Operation *module) {
  std::vector<mlir::FunctionOpInterface> funcList;
  DenseMap<SymbolRefAttr, LIT::StructDeclOp> structMap;
  module->walk([&](Operation *op) {
    // Collect functions and nested functions.
    if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op))
      funcList.push_back(funcOp);
    // Collect structs.
    if (auto structOp = dyn_cast<LIT::StructDeclOp>(op))
      structMap[getFullyResolvedSymbolRef(structOp)] = structOp;
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

  /// Given a field number that indicates a stored field in the specified type,
  /// return the name of the field that contains it as well as its declared
  /// type.
  std::pair<StringAttr, Type> getFieldContaining(DeclRefType type,
                                                 unsigned fieldNo);

  /// Return the struct decl for the specified DeclRefType.
  LIT::StructDeclOp getStructDeclForType(DeclRefType type) const {
    // If not, we compute it recursively.  Structs cannot be infinitely deep, so
    // we can just do this recursively.
    auto it = structMap.find(type.getSymbol());
    assert(it != structMap.end() && "reference to struct that wasn't declared");
    return it->second;
  }

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
  auto smIt = structMap.find(structSymbol);
  assert(smIt != structMap.end() && smIt->second &&
         "reference to struct that wasn't declared");
  LIT::StructDeclOp decl = smIt->second;

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

/// Given a field number that indicates a stored field in the specified type,
/// return the name of the field that contains it as well as its declared
/// type.
std::pair<StringAttr, Type>
TypeDeclInfo::getFieldContaining(DeclRefType declRef, unsigned fieldNo) {
  LIT::StructDeclOp decl = getStructDeclForType(declRef);

  // Scan to find the field that contains this.
  unsigned startFieldIdx = 0;
  for (auto field : decl.getFieldDecls()) {
    // This range check is needed to handle zero-sized fields: they don't
    // contain a field even if they start at the beginning of it.
    unsigned numSubFields = getNumFieldsInType(field.getType());
    if (startFieldIdx <= fieldNo && startFieldIdx + numSubFields > fieldNo)
      return {field.getNameAttr(), field.getType()};
    startFieldIdx += numSubFields;
  }

  llvm_unreachable("invalid index into struct field numbering");
}

//===----------------------------------------------------------------------===//
// LifetimeTrackable
//===----------------------------------------------------------------------===//

/// This class provide an abstraction for analyzing lifetime-trackable values,
/// e.g. variable definitions and owned arguments to functions.
struct LifetimeTrackable {
  LifetimeTrackable(Value value);

  operator bool() const { return name != StringAttr(); }

  /// This is the user's declared name for the value declaration, or null if
  /// this isn't a tracked value.
  StringAttr name;

  /// This is true if the SSA value is a pointer to the logical storage instead
  /// of being the value itself.  This is always true for values of memory-only
  /// type.
  bool isIndirect = false;

  /// This is true if the value is uninitialized at function entry, false if it
  /// starts out initialized.
  bool startsUninit = false;

  /// This is true if the value is uninitialized at function exist, false if it
  /// ends up defined (e.g. as with a byref argument).
  bool endsUninit = false;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  bool isFullObjectLiveOnEntry = false;

  /// Return the type of the underlying value, looking through the pointer type
  /// if this is an indirect reference.
  Type getValueType(Value value) const;

  /// Get the destructor for this value if one exists, otherwise return null.
  TypedAttr getDestructor(Value value, TypeDeclInfo &typeDeclInfo) const;
};

LifetimeTrackable::LifetimeTrackable(Value v) {
  if (!v) // Null value isn't tracked.
    return;

  // LetReg starts out initialized with its own value.
  if (auto letReg = v.getDefiningOp<LetRegDeclOp>()) {
    name = letReg.getNameAttr();
    isIndirect = false;
    startsUninit = false;
    endsUninit = true;
    return;
  }
  // VarLetDeclOp is uninit and ends that way.
  if (auto varLet = v.getDefiningOp<VarLetDeclOp>()) {
    name = varLet.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  /// Owned results of function calls are tracked as being initialized when
  /// defined but needing to be destroyed by the end of function.
  if (OpResult res = dyn_cast<OpResult>(v)) {
    if (auto call = dyn_cast<KGENCallOpInterface>(res.getOwner())) {
      if (call.getCalleeType().hasOwnedRegisterResult() &&
          !isa<AddressOfOp>(res.getOwner())) {
        name = StringAttr::get(v.getContext(), "<call result>");
        isIndirect = false;
        startsUninit = false;
        endsUninit = true;
      }
    }
  }

  // If this is a function argument, check to see what ownership it has.
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg || !bbArg.getOwner())
    return;
  SignatureType signature;
  Operation *parentOp = bbArg.getOwner()->getParentOp();
  StringArrayAttr valueNames;
  if (auto func = dyn_cast<LIT::FuncOp>(parentOp)) {
    signature = func.getSignature();
    valueNames = func.getValueParamNamesAttr();
  } else if (auto func = dyn_cast<ParamDeclareRegionOp>(parentOp)) {
    signature = func.getSignature();
    // FIXME(Issue #11918): Need valueNames for nested functions.
  } else
    return;

  switch (signature.getValueInputConventions()[bbArg.getArgNumber()]) {
  case ValueInputConvention::OwnedInReg: // This gets and LValue slot.
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::BorrowedInMem:
    // These are immutable so don't need to be tracked.
    return;

  case ValueInputConvention::OwnedInMem:
    isIndirect = true;
    startsUninit = false;
    endsUninit = true;
    break;
  case ValueInputConvention::ByRefResult:
    // FIXME(Issue#12196): __result__ slots in raising functions cannot properly
    // model the behavior when an error is thrown, so we give up tracking them.
    if (signature.isThrows())
      return;
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    break;
  case ValueInputConvention::InitSelf:
    // Unlike byref-result, we allow memberwise initialization of 'self' in an
    // init method to construct a full value.
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    isFullObjectLiveOnEntry = true;
    break;
  case ValueInputConvention::ByRef:
    isIndirect = true;
    startsUninit = false;
    endsUninit = false;
    break;
  }

  // FIXME(Issue #11918): Need valueNames for nested functions.
  name = valueNames
             ? valueNames[bbArg.getArgNumber()]
             : StringAttr::get(bbArg.getContext(), "FIXME(Issue #11918)");
}

/// Return the type of the underlying value, looking through the pointer type
/// if this is an indirect reference.
Type LifetimeTrackable::getValueType(Value value) const {
  // If this is a direct value, use the type directly.
  if (!isIndirect)
    return value.getType();

  auto pointee = llvm::cast<POP::PointerType>(value.getType()).getElementType();
  if (auto type = dyn_cast<TypeConstantAttr>(pointee))
    return type.getValue();
  return ParamRefType::get(pointee);
}

/// Get the destructor for this value if one exists, otherwise return null.
TypedAttr LifetimeTrackable::getDestructor(Value value,
                                           TypeDeclInfo &typeDeclInfo) const {
  DeclRefType valueType = dyn_cast<DeclRefType>(getValueType(value));
  if (!valueType)
    return {};
  return typeDeclInfo.getStructDeclForType(valueType).getDestructorAttr();
}

//===----------------------------------------------------------------------===//
// ValueInfo / ValueSet tracking
//===----------------------------------------------------------------------===//

namespace {
struct ValueInfo {
  /// This is the declared value being tracked.
  const Value value;

  /// This indicates the (first, end] bitrange in the bit vector corresponding
  /// to this value.
  const unsigned startValueBit, endValueBit;

  /// True if this values starts out uninitialized at the beginning of its
  /// lifetime.
  const bool startsUninit;
  /// True if this value needs to be uninitialized at the end of its lifetime.
  const bool endsUninit;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  const bool isFullObjectLiveOnEntry;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;
};

/// A ValueRef indicates a slice reference into the BitVector for all the
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

  /// Test if all the bits in the range are set in the specified BitVector.
  bool isAllPresent(const BitVector &bits) const {
    // BitVector doesn't have a more efficient method for this.  We could make
    // this more efficient for longer ranges if needed.
    for (size_t i = startBit, e = endBit; i != e; ++i)
      if (!bits[i])
        return false;
    return true;
  }

  /// Test if all the bits in the range are clear in the specified BitVector.
  bool isAllMissing(const BitVector &bits) const {
    // BitVector doesn't have a more efficient method for this.  We could make
    // this more efficient for longer ranges if needed.
    for (size_t i = startBit, e = endBit; i != e; ++i)
      if (bits[i])
        return false;
    return true;
  }

  /// Set the bits for this range to zero or one in the specified BitVector.
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
    addValue(Value(), LifetimeTrackable(Value()));
  }

  /// Return the number of values we are tracking.
  MutableArrayRef<ValueInfo> getValueInfos() { return valueInfos; }
  ValueInfo &getValueInfo(size_t idx) { return valueInfos[idx]; }
  const ValueInfo &getValueInfo(size_t idx) const { return valueInfos[idx]; }

  /// Add a value to the set that we are tracking.  This includes:
  ///  * the MLIR representation for the value itself
  ///  * whether the value is a by-ref pointer to the underlying logical value
  ///  * the destructor for the value
  ///  * whether the value starts out uninit or init at the function start
  ///  * whether the value is uninit or init at normal function return.
  ///
  void addValue(Value val, LifetimeTrackable trackable) {
    unsigned firstValueBit = getNumTotalBits();
    unsigned numValueBits = 1;
    // We are only field sensitive for memory objects, not in-register values.
    // It isn't possible to update in-register values: register_passable values
    // are always valid when present.  If they pass through memory, they are
    // checked when loaded from memory.
    if (val && trackable.isIndirect) {
      Type valType = trackable.getValueType(val);
      // We track one extra bit for the value so we know if it is fully
      // initialized or not.  If so, we can run the destructor on the entire
      // aggregate when it is unused.  This also allows us to track the
      // initialization state of structures with no fields.
      numValueBits = typeDeclInfo.getNumFieldsInType(valType) + 1;
    }

    valueInfoIndex[val] = valueInfos.size();
    valueInfos.push_back({val, firstValueBit, firstValueBit + numValueBits,
                          trackable.startsUninit, trackable.endsUninit,
                          trackable.isFullObjectLiveOnEntry,
                          /*hasErrorDiagnosed=*/false});
  }

  /// Return a reference to the entire value with the specified ID.
  ValueRef getFullValueRef(unsigned valueId) const {
    const auto &entry = valueInfos[valueId];
    return ValueRef{valueId, entry.startValueBit, entry.endValueBit};
  }

  /// If this value is directly tracked by the ValueSet, return the index of the
  /// value, otherwise return zero.
  ValueRef getDirectValueRef(Value value) const {
    auto it = valueInfoIndex.find(value);
    if (it == valueInfoIndex.end())
      return ValueRef();
    return getFullValueRef(it->second);
  }

  /// Given a pointer that is being accessed indirectly by an operation, return
  /// the value number being referenced, or zero if not tracked.
  ValueRef getPointerValueIndex(Value value);

  /// Return the total number of bits we need to track in the bitvector.
  unsigned getNumTotalBits() const {
    return !valueInfos.empty() ? valueInfos.back().endValueBit : 0;
  }

  TypeDeclInfo &typeDeclInfo;

private:
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

  void scanFunction(mlir::FunctionOpInterface func);
  void scanBlock(Block &body);

private:
  void checkOp(Operation &op);
  void checkLive(Operation &op, ValueRef valueRef);

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

static Type digIntoTypeAtFieldOffset(Type type, unsigned firstInvalidOffset,
                                     unsigned nextValidOffset,
                                     InFlightDiagnostic &diag,
                                     TypeDeclInfo &typeDeclInfo) {
  // Dig into the type to get to the right field.
  while (firstInvalidOffset) {
    // To index into this type, it must be a DeclRef.
    DeclRefType declRefType = cast<DeclRefType>(type);
    auto [fieldName, fieldType] =
        typeDeclInfo.getFieldContaining(declRefType, firstInvalidOffset);
    unsigned fieldBitOffset =
        typeDeclInfo.getFieldIndex(declRefType, fieldName);
    firstInvalidOffset -= fieldBitOffset;
    nextValidOffset -= fieldBitOffset;
    type = fieldType;
    diag << "." << fieldName.str();
  }

  // Dig into the field to ignore trailing members that we don't care about.
  while (nextValidOffset < typeDeclInfo.getNumFieldsInType(type)) {
    DeclRefType declRefType = cast<DeclRefType>(type);
    auto [fieldName, fieldType] =
        typeDeclInfo.getFieldContaining(declRefType, 0);
    type = fieldType;
    diag << "." << fieldName.str();
  }

  return type;
}

/// When complaining about a specific value, check to see if the /entire/
/// field-sensitive value is missing from the specified bitvector.  If not,
/// add a suffix that identifies the first whole field that is missing.
static void addBadValueNameToDiag(const LifetimeTrackable &trackable,
                                  ValueRef valueRef, const BitVector &bits,
                                  ValueSet &valueSet,
                                  mlir::InFlightDiagnostic &diag) {
  diag << "'" << trackable.name.str();
  // If the whole value is missing, then don't add any field information.
  if (valueRef.isAllMissing(bits)) {
    diag << "'";
    return;
  }

  // Figure out what the end of the field bits are so we can report the first
  // fields.  The full object ends with a bit to track whether the whole value
  // is initialized which we don't want to track.
  const ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);
  unsigned fullValueStartBit = valueEntry.startValueBit;

  unsigned endOfFullObjectFields = valueEntry.endValueBit - 1;
  if (endOfFullObjectFields == fullValueStartBit) {
    // No stored fields!
    diag << "'";
    return;
  }

  // The end of the reference is either the end of valueref (if that was a
  // subfield of the overall object) or it is the end of full object.
  unsigned endOfAccessFields = std::min(endOfFullObjectFields, valueRef.endBit);

  // We know that something in valueRef is missing, but we don't know which
  // piece.  Find the first bit in valueRef that isn't live.
  unsigned firstMissingFieldNo =
      std::min(unsigned(bits.find_next_unset(valueRef.startBit - 1U)),
               endOfAccessFields - 1);
  // Find the area of overlap so we complain about larger aggregates that are
  // fully uninit, not tiny parts of them.
  unsigned firstPresentFieldNo = std::min(
      unsigned(bits.find_next(firstMissingFieldNo)), endOfAccessFields);

  // Ok, the uninitialized thing is [firstMissingFieldNo, firstPresentFieldNo)
  // so we want to figure out which sub-piece of the whole value type is the
  // problem, and identify a path that drills down through each of the named
  // fields.
  auto type = trackable.getValueType(valueEntry.value);
  // Emit the field prefix for the specified type.
  digIntoTypeAtFieldOffset(type, firstMissingFieldNo - fullValueStartBit,
                           firstPresentFieldNo - fullValueStartBit, diag,
                           valueSet.typeDeclInfo);
  diag << "'";
}

// Verify that the specified ValueRef is live at this point, diagnosing an
// error at the specified operation if not.
void UninitializedValueScan::checkLive(Operation &op, ValueRef valueRef) {
  // If the value is live then all is good.
  if (valueRef.isAllPresent(liveValues))
    return;

  // Ok, it isn't, gear up to see how to best report the error.
  ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);
  if (valueEntry.hasErrorDiagnosed)
    return; // Only report one error per symbolic value.
  valueEntry.hasErrorDiagnosed = true;

  LifetimeTrackable trackable(valueEntry.value);
  assert(trackable && "value should be tracked");

  // If the fields are all valid except for the whole-object bit, then the user
  // tried to initialize a value by initializing all its fields.  Reject this
  // with a customized error.
  if (trackable.isIndirect && valueRef.endBit == valueEntry.endValueBit &&
      ValueRef(valueRef.valueId, valueRef.startBit, valueRef.endBit - 1)
          .isAllPresent(liveValues)) {
    auto diag = mlir::emitError(op.getLoc(), "'")
                << trackable.name.str()
                << "' used with all fields manually initialized "
                   "but without calling an '__init__' method";
    return;
  }

  auto diag = mlir::emitError(op.getLoc(), "use of uninitialized value ");

  // If some fields are present and others are missing, complain about the first
  // whole field that is missing.
  addBadValueNameToDiag(trackable, valueRef, liveValues, valueSet, diag);

  diag.attachNote(valueEntry.value.getLoc())
      << "'" << trackable.name.str() << "' declared here";
}

void UninitializedValueScan::scanFunction(mlir::FunctionOpInterface func) {
  // Initialize the BitVector with all the elements that are live-in.  We treat
  // all values live at the start of the function (even before they are actually
  // defined) because we know that all uses must be after them due to SSA
  // dominance.
  liveValues.resize(valueSet.getNumTotalBits());
  for (const ValueInfo &valueInfo : valueSet.getValueInfos())
    if (!valueInfo.startsUninit) {
      // If the whole value is live on entry, notice that.
      liveValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
    } else if (valueInfo.isFullObjectLiveOnEntry) {
      // If /just/ the full object bit is live on entry, set it.
      liveValues.set(valueInfo.endValueBit - 1);
    }

  // Scan the body of the function.
  scanBlock(func.getFunctionBody().front());
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

  auto checkSSAValueLive = [&](Value value) -> ValueRef {
    ValueRef valueId = valueSet.getDirectValueRef(value);
    if (valueId)
      checkLive(op, valueId);
    return valueId;
  };

  auto checkDirectPointerLive = [&](Value value) -> ValueRef {
    ValueRef valueRef = valueSet.getPointerValueIndex(value);
    if (valueRef)
      checkLive(op, valueRef);
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
    checkLive(op, valueSet.getPointerValueIndex(loadOp.getPtr()));
    return;
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
    SignatureType signature = call.getCalleeType();
    ValueRange operands;
    if (isa<CallOp, CallParamOp, AsyncCallOp>(op))
      operands = call->getOperands();
    else if (isa<POP::CallIndirectOp>(op))
      operands = call->getOperands().drop_front();
    else {
      assert(isa<AddressOfOp>(op) && "Unknown call op");
      return; // AddressOf isn't a use of any SSA values.
    }

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
      case ValueInputConvention::InitSelf:
        // This call defines the by-ref result.
        valueSet.getPointerValueIndex(operand).markBits(liveValues, true);
        break;
      }
    }

    // If the result is defining an owned register value, then we treat this as
    // a definition.
    if (signature.hasOwnedRegisterResult())
      valueSet.getDirectValueRef(op.getResult(0)).markBits(liveValues, true);

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
        checkLive(op, valueSet.getFullValueRef(i));
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
// DestructorInsertion
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the third pass over a function body, which
/// inserts destructors after the last use of values.
struct DestructorInsertion {
  DestructorInsertion(ValueSet &valueSet) : valueSet(valueSet) {}
  DestructorInsertion(const DestructorInsertion &existing) = delete;

  void scanFunction(mlir::FunctionOpInterface func);
  void scanBlock(Block &body);

private:
  void checkOp(Operation &op);
  void markConsumed(ValueRef valueRef);
  void checkLive(Operation &op, ValueRef valueRef);
  void checkUse(Operation &op, ValueRef valueRef);
  void checkDef(Operation &op, ValueRef valueRef);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is the set of values known to be used below this point, so they
  /// should not be destroyed if there are uses.  Any use of a value /not/ in
  /// this set will be a last use that does get destroyed.
  BitVector consumedValues;
};
} // namespace

void DestructorInsertion::scanFunction(mlir::FunctionOpInterface func) {
  consumedValues.resize(valueSet.getNumTotalBits());

  // Scan the body of the function.
  scanBlock(func.getFunctionBody().front());
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void DestructorInsertion::scanBlock(Block &block) {
  for (Operation &op : llvm::reverse(block))
    checkOp(op);
}

void DestructorInsertion::checkOp(Operation &op) {
  // This op is handled when used.
  if (isa<StructGEPOp>(op))
    return;

  // A store to a value is an overwrite - this means that any incoming values
  // are unused and should be destroyed if they exist.
  if (auto storeOp = dyn_cast<POP::StoreOp>(op)) {
    checkDef(op, valueSet.getPointerValueIndex(storeOp.getPtr()));
    return;
  }

  // A load is a use of whatever fields are being referenced.  If this is the
  // /last/ use of a value, emit a destructor of that value.
  if (auto loadOp = dyn_cast<POP::LoadOp>(op)) {
    checkUse(op, valueSet.getPointerValueIndex(loadOp.getPtr()));
    return;
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
    SignatureType signature = call.getCalleeType();
    ValueRange operands;
    if (isa<CallOp, CallParamOp, AsyncCallOp>(op))
      operands = call->getOperands();
    else if (isa<POP::CallIndirectOp>(op))
      operands = call->getOperands().drop_front();
    else {
      assert(isa<AddressOfOp>(op) && "Unknown call op");
      return; // AddressOf isn't a use of any SSA values.
    }

    // If the result is defining an owned register value, treat it as a def.
    if (signature.hasOwnedRegisterResult())
      checkDef(op, valueSet.getDirectValueRef(op.getResult(0)));

    assert(signature.getValueInputConventions().size() == operands.size());
    for (auto [convention, operand] :
         llvm::zip(signature.getValueInputConventions(), operands)) {
      switch (convention) {
      case ValueInputConvention::OwnedInReg:
        // This consumes the value, so it isn't dead going upwards.
        valueSet.getDirectValueRef(operand).markBits(consumedValues, true);
        break;
      case ValueInputConvention::OwnedInMem:
        valueSet.getPointerValueIndex(operand).markBits(consumedValues, true);
        break;
      case ValueInputConvention::BorrowedInReg:
        checkUse(op, valueSet.getDirectValueRef(operand));
        break;
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::ByRef:
        checkUse(op, valueSet.getPointerValueIndex(operand));
        break;
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        // This defines the memory it writes to.
        checkDef(op, valueSet.getPointerValueIndex(operand));
        break;
      }
    }
    return;
  }

  // LetReg takes ownership of its operand value and defines its own value.
  if (auto letReg = dyn_cast<LetRegDeclOp>(op)) {
    // This defines the result value.  Emit a destructor if unused.
    checkDef(op, valueSet.getDirectValueRef(letReg));
    // This consumes the input.
    markConsumed(valueSet.getDirectValueRef(letReg.getOperand()));
    return;
  }

  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp>(op)) {
    for (const ValueInfo &valueInfo : valueSet.getValueInfos())
      if (!valueInfo.endsUninit)
        consumedValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
    return;
  }

  // A unreachable consumes everything.  Nothing needs to be destroyed here!
  if (isa<UnreachableOp>(op)) {
    consumedValues.set();
    return;
  }

  // Otherwise this some other operation that using a SSA value.  If this is the
  // last use of the value, make sure we destroy it when done.
  for (Value operand : op.getOperands())
    checkUse(op, valueSet.getDirectValueRef(operand));
}

// When the specified value is consumed by an operation we know it doesn't need
// to be destroyed above this point.
void DestructorInsertion::markConsumed(ValueRef valueRef) {
  valueRef.markBits(consumedValues, true);
}

/// This operation uses whatever fields are being referenced.  Iff this is the
/// /last/ use of a value, emit a destructor of the overall value.
void DestructorInsertion::checkUse(Operation &op, ValueRef valueRef) {
  // If this is an untracked reference or if all the referenced data is already
  // known to be consumed at this point, then there is nothing to do.
  if (!valueRef || valueRef.isAllPresent(consumedValues))
    return;

  ValueInfo &valueInfo = valueSet.getValueInfos()[valueRef.valueId];
  if (valueInfo.hasErrorDiagnosed)
    return;

  LifetimeTrackable trackable(valueInfo.value);
  assert(trackable && "shouldn't be tracking untrackable things!");

  // Otherwise it is a use happening when the values become dead, we need to
  // emit a destructor call.  Check to make sure they whole value dies at once.
  ValueRef fullValueRef = valueSet.getFullValueRef(valueRef.valueId);

  TypedAttr dtor =
      trackable.getDestructor(valueInfo.value, valueSet.typeDeclInfo);

  if (!dtor)
    return;

  // FIXME: Move this after checking below.
  if (!fullValueRef.isAllMissing(consumedValues)) {
    auto diag = mlir::emitError(
        op.getLoc(), "last use of value with partially dead members ");
    // Identify the first sub-field that is missing.
    addBadValueNameToDiag(trackable, fullValueRef, consumedValues, valueSet,
                          diag);
    valueInfo.hasErrorDiagnosed = true;
    return;
  }

  // Ok, this value will be consumed by the dtor we are emitting, so any uses
  // above this point won't need to emit this.
  fullValueRef.markBits(consumedValues, true);

  SignatureType signature = cast<SignatureType>(dtor.getType());
  assert(signature.getValueResults().size() == 1 &&
         "dtor should have one result (none type)");
  assert(signature.getValueInputs().size() == 1 &&
         "dtor should have one operand");

  OpBuilder b(&op);
  b.setInsertionPointAfter(&op);

  // We may have the value indirect (e.g. because it is in a var) which needs
  // to be loaded to invoke the destructor.
  Value valueToDestroy = valueInfo.value;
  if (valueToDestroy.getType() != signature.getValueInputs()[0]) {
    assert(POP::PointerType::get(signature.getValueInputs()[0]) ==
           valueToDestroy.getType());
    valueToDestroy = b.create<POP::LoadOp>(op.getLoc(), valueToDestroy,
                                           /*align*/ std::nullopt);
  }

  // Emit the call to the destructor.
  if (auto symbolConstantDtor = dyn_cast<SymbolConstantAttr>(dtor)) {
    b.create<CallOp>(op.getLoc(), signature.getValueResults()[0],
                     symbolConstantDtor,
                     b.getAttr<ParamDeclArrayAttr>(ArrayRef<ParamDeclAttr>()),
                     ValueRange(valueToDestroy));
  } else {
    b.create<CallParamOp>(
        op.getLoc(), signature.getValueResults()[0], dtor,
        b.getAttr<ParamDeclArrayAttr>(ArrayRef<ParamDeclAttr>()),
        ValueRange(valueToDestroy));
  }
}

/// This operation defines the specified value.  If the value is dead on
/// arrival, emit a destructor of the value.
void DestructorInsertion::checkDef(Operation &op, ValueRef valueRef) {
  // If there is no use of the value being defined, emit a dtor after the op.
  checkUse(op, valueRef);

  // This call defines the result, so anything above it is either dead or
  // needs a destructor if live.
  valueRef.markBits(consumedValues, false);
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

  LogicalResult processFunction(mlir::FunctionOpInterface func,
                                TypeDeclInfo &typeDeclInfo);
};
} // namespace

LogicalResult CheckLifetimes::processFunction(mlir::FunctionOpInterface func,
                                              TypeDeclInfo &typeDeclInfo) {
  // Pass #1: Collect all of the values declared in the function that have
  // ownership to track, and number them.
  ValueSet valueSet(typeDeclInfo);
  func->walk([&](Operation *op) -> WalkResult {
    // Skip looking at nested functions, they are handled as separate contexts.
    if (isa<ParamDeclareRegionOp>(op) && op != func)
      return WalkResult::skip();

    // All the ops that define trackable values have a single result.
    if (op->getNumResults() == 1)
      if (auto trackable = LifetimeTrackable(op->getResult(0)))
        valueSet.addValue(op->getResult(0), trackable);

    // If there are any regions, check the block arguments for arguments.
    for (auto &region : op->getRegions()) {
      for (auto &block : region)
        for (auto arg : block.getArguments())
          if (auto trackable = LifetimeTrackable(arg))
            valueSet.addValue(arg, trackable);
    }

    return WalkResult::advance();
  });

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined, emitting diagnostics as we go.
  UninitializedValueScan(valueSet).scanFunction(func);

  // TODO: How do we want to handle captures in closures?  Their uses
  // effectively form the capture list for the closure.  Should this get
  // materialized by LowerSemanticCF before this pass?

  DestructorInsertion(valueSet).scanFunction(func);

  // Return failure if we generated errors for any of the tracked values.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  }));
}
