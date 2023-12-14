//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass checks value lifetime invariants, e.g. that
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

/// Find all the functions and types in the module.
static std::tuple<std::vector<LIT::FuncOp>,
                  DenseMap<SymbolRefAttr, LIT::FuncOp>,
                  DenseMap<SymbolRefAttr, LIT::StructDeclOp>,
                  DenseMap<SymbolRefAttr, LIT::TraitDeclOp>>
collectFunctionsAndTypes(Operation *module) {
  std::vector<LIT::FuncOp> funcList;
  DenseMap<SymbolRefAttr, LIT::FuncOp> funcMap;
  DenseMap<SymbolRefAttr, LIT::StructDeclOp> structMap;
  DenseMap<SymbolRefAttr, LIT::TraitDeclOp> traitMap;
  module->walk([&](Operation *op) {
    // Collect functions and nested functions.
    if (auto funcOp = dyn_cast<LIT::FuncOp>(op)) {
      funcMap[getFullyResolvedSymbolRef(funcOp)] = funcOp;

      // We don't process external functions. They don't have a body to check.
      if (funcOp.isExternal())
        return;
      funcList.push_back(funcOp);
    }
    // Collect structs.
    else if (auto structOp = dyn_cast<LIT::StructDeclOp>(op)) {
      structMap[getFullyResolvedSymbolRef(structOp)] = structOp;
    } else if (auto traitOp = dyn_cast<LIT::TraitDeclOp>(op)) {
      traitMap[getFullyResolvedSymbolRef(traitOp)] = traitOp;
    }
  });
  return {std::move(funcList), std::move(funcMap), std::move(structMap),
          std::move(traitMap)};
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
  TypeDeclInfo(DenseMap<SymbolRefAttr, LIT::StructDeclOp> &&structMap,
               DenseMap<SymbolRefAttr, LIT::FuncOp> &&funcMap,
               DenseMap<SymbolRefAttr, LIT::TraitDeclOp> &&traitMap)
      : structMap(std::move(structMap)), funcMap(std::move(funcMap)),
        traitMap(std::move(traitMap)) {}

  /// Return the total number of flattened fields in the specified type.
  unsigned getNumFieldsInType(Type type);

  /// Return the start bit for a field with the specified name in the specified
  /// type.
  unsigned getFieldIndex(DeclRefType type, StringAttr fieldName) const;

  /// Given a field number that indicates a stored field in the specified type,
  /// return the name of the field that contains it as well as its declared
  /// type.
  std::pair<StringAttr, Type> getFieldContaining(DeclRefType type,
                                                 unsigned fieldNo);

  /// Return the struct decl for the specified DeclRefType.
  LIT::StructDeclOp getStructDeclForType(DeclRefType type) const {
    auto it = structMap.find(type.getSymbol());
    assert(it != structMap.end() && "reference to struct that wasn't declared");
    return it->second;
  }

  /// Return the trait decl for the specified TraitType.
  LIT::TraitDeclOp getTraitDeclForType(TraitType type) const {
    auto it = traitMap.find(type.getSymbol());
    assert(it != traitMap.end() && "reference to trait that wasn't declared");
    return it->second;
  }

  /// Return true if the specified type is RegisterPassableTrivial - no copy,
  /// move, or destructor members.
  bool isRegisterPassableTrivial(Type type) const;

  /// Given the RValue type for a value that needs to be destroyed, return the
  /// destructor the invoke, or null if there is none.
  TypedAttr getDestructorForType(Type type) const;
  SymbolConstantAttr getMoveInitForType(Type type) const;

  /// Return the function for a given symbol name if known.
  LIT::FuncOp getFuncForSymbol(SymbolRefAttr symbolRef) const {
    auto it = funcMap.find(symbolRef);
    return it != funcMap.end() ? it->second : LIT::FuncOp();
  }

private:
  DenseMap<SymbolRefAttr, StructDeclOp> structMap;
  DenseMap<SymbolRefAttr, LIT::FuncOp> funcMap;
  DenseMap<SymbolRefAttr, TraitDeclOp> traitMap;

  /// This keeps track of the number of fields in the struct specified by the
  /// (fully flattened) symbol and parameters.
  DenseMap<DeclRefType, unsigned> numFields;

  /// A map from struct name and field name to index within the struct.  This
  /// isn't the field number, this is the number of recursively flattened
  /// fields until the start of the field.
  DenseMap<std::pair<SymbolRefAttr, StringAttr>, unsigned> fieldIndices;
};

/// Return true if the specified type is RegisterPassableTrivial - no copy,
/// move, or destructor members.
bool TypeDeclInfo::isRegisterPassableTrivial(Type type) const {
  DeclRefType valueType = dyn_cast<DeclRefType>(type);
  if (!valueType) // Values of raw MLIR type are always trivial.
    return true;
  return getStructDeclForType(valueType).isRegisterPassableTrivial();
}

static SymbolConstantAttr getSpecialMemberForType(
    Type type, const TypeDeclInfo *typeDecls,
    llvm::function_ref<SymbolConstantAttr(StructDeclOp)> getMember) {
  auto valueType = dyn_cast<DeclRefType>(type);
  if (!valueType) // Values of raw MLIR type don't have destructors.
    return {};
  SymbolConstantAttr attr =
      getMember(typeDecls->getStructDeclForType(valueType));
  if (!attr)
    return {};

  // If there are parameters to the type, then the dtor will have those
  // parameters as well, substitute them in.
  assert(attr.getParamValues().empty() && "dtor should be unparameterized");
  if (valueType.getParamValues().empty())
    return attr;

  ArrayRef<TypedAttr> paramValues = valueType.getParamValues();
  auto newSig = attr.getType().getSpecializedSignature(
      paramValues, []() -> InFlightDiagnostic {
        assert(false && "getSpecializedSignature should not error here");
        return {};
      });
  return SymbolConstantAttr::get(attr.getSymbol(), paramValues, newSig);
}

/// Given the RValue type for a value that needs to be destroyed, return the
/// destructor the invoke, or null if there is none.
TypedAttr TypeDeclInfo::getDestructorForType(Type type) const {
  if (auto generic = dyn_cast<ParamRefType>(type)) {
    if (auto trait = dyn_cast<TraitType>(generic.getParam().getType())) {
      SignatureType dtorSig = TraitDeclOp(traitMap.at(trait.getSymbol()))
                                  .getDtorSig()
                                  .value_or(SignatureType());
      if (dtorSig) {
        return ParamOperatorAttr::get(
            POC::GetTypeMethod,
            {generic.getParam(),
             StringAttr::get("__del__", StringType::get(type.getContext()))},
            dtorSig.getSpecializedSignature(
                {TypeConstantAttr::get(trait,
                                       AnyRegTypeType::get(type.getContext())),
                 generic.getParam()},
                []() -> InFlightDiagnostic {
                  assert(false &&
                         "getSpecializedSignature not expected to fail here");
                  return {};
                }));
      }
    }
  }

  return getSpecialMemberForType(type, this, [](StructDeclOp structOp) {
    return structOp.getDestructorAttr();
  });
}

SymbolConstantAttr TypeDeclInfo::getMoveInitForType(Type type) const {
  return getSpecialMemberForType(type, this, [](StructDeclOp structOp) {
    return structOp.getMoveInitAttr();
  });
}

/// Return the total number of flattened fields in the specified type.
unsigned TypeDeclInfo::getNumFieldsInType(Type type) {
  // We currently treat all non-struct types as being a single element, even
  // things like kgen.list containing struct types.
  DeclRefType declRef = dyn_cast<DeclRefType>(type);
  if (!declRef)
    return 1;

  // See if we've already looked this up, if so, just return the known value.
  auto it = numFields.find(declRef);
  if (it != numFields.end())
    return it->second;

  // If not, we compute it recursively.  Structs cannot be infinitely deep, so
  // we can just do this recursively.
  SymbolRefAttr structSymbol = declRef.getSymbol();
  auto smIt = structMap.find(structSymbol);
  assert(smIt != structMap.end() && smIt->second &&
         "reference to struct that wasn't declared");
  LIT::StructDeclOp decl = smIt->second;

  // Initialize a parameter evaluator. We need to compute the resolved field
  // types to recursively compute the number of fields.
  ParameterEvaluator evaluator;
  for (auto [decl, value] :
       llvm::zip(decl.getInputParams(), declRef.getParamValues()))
    evaluator.setParameterValue(decl, value);

  size_t totalFields = 0;
  for (auto field : decl.getFieldDecls()) {
    fieldIndices[{structSymbol, field.getNameAttr()}] = totalFields;
    totalFields +=
        getNumFieldsInType(evaluator.getReboundType(field.getType()));
  }

  // We always track an extra bit per struct.  On the outer level of a value
  // this tracks whether the object is fully constructed (not just field
  // constructed).  On individual fields, it tracks whether the field itself is
  // initialized or whether its subfields are initialized.  This also allows us
  // to support (sub)fields that have zero members soundly.
  ++totalFields;

  return numFields[declRef] = totalFields;
}

/// Return the start bit for a field with the specified name in the specified
/// type.
unsigned TypeDeclInfo::getFieldIndex(DeclRefType type,
                                     StringAttr fieldName) const {
  auto it = fieldIndices.find({type.getSymbol(), fieldName});
  assert(it != fieldIndices.end() &&
         "shouldn't get field index of unused value");
  return it->second;
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

  /// True if this value lives in memory, not a @register_passable SSA value.
  const bool isIndirect;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  const bool isFullObjectLiveOnEntry;

  /// True if this is a 'let' declaration which isn't allowed to be mutated
  /// after it is initialized.
  const bool isLet;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;

  /// This bit gets set to true when an already-initialized var gets
  /// overwritten.  At the end of the uninit analysis, vars that are not mutated
  /// get a warning that indicate they should be written as 'let's.  This is
  /// also used to implement late initialization for lets.
  bool isMutatedWhenInitialized;

  /// Return true if this value contains the specified bit.
  bool contains(unsigned bitNo) const {
    return startValueBit <= bitNo && bitNo < endValueBit;
  }

  StringAttr getName() const {
    assert(value && "cannot get name of null entry");
    return LifetimeTrackable(value).name;
  }
};

/// A ValueRef indicates a slice reference into the BitVector for all the
/// values.
struct ValueRef {
  /// This is the entry # for the ValueInfo for the overall value.
  unsigned valueId = 0;

  /// This is the (start, end] span of bits for the reference that we're
  /// tracking, which may be a subset of the overall value.
  unsigned startBit = ~0U, endBit = ~0U;

  /// This is true if this value reference is looking at the value indirectly,
  /// not as a @register_passable value in an SSA value.
  bool isIndirect = false;

  ValueRef() = default;
  ValueRef(unsigned valueId, unsigned startBit, unsigned endBit,
           bool isIndirect)
      : valueId(valueId), startBit(startBit), endBit(endBit),
        isIndirect(isIndirect) {}

  /// Allow use of a ValueRef in a boolean condition.
  operator bool() const { return valueId != 0; }

  unsigned getNumBits() const { return endBit - startBit; }

  bool operator==(ValueRef rhs) const {
    return startBit == rhs.startBit && endBit == rhs.endBit;
  }
  bool operator!=(ValueRef rhs) const { return !(*this == rhs); }

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

  /// Return the type of the underlying value, looking through the pointer type
  /// if this is an indirect reference.
  Type getValueType(Value value) const {
    return LifetimeTrackable::getTypeOrPointeeType(value.getType(), isIndirect);
  }

  /// Given a field ref with fields, return a sub-field that starts at the
  /// specified bit offset and has the specified size.
  ValueRef getSubfield(unsigned offset, unsigned width) const {
    assert(startBit + offset + width <= endBit && "Not a valid subfield");
    return ValueRef(valueId, startBit + offset, startBit + offset + width,
                    isIndirect);
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
  /// This provides information about the types referenced from values, e.g. the
  /// number of fields they have.
  TypeDeclInfo &typeDeclInfo;

  /// Initialize the value set with one entry, so index #0 is always invalid and
  /// can be used as a sentinel, and so a null Value is always treated as
  /// untracked.
  ValueSet(TypeDeclInfo &typeDeclInfo, LIT::FuncOp func)
      : typeDeclInfo(typeDeclInfo), func(func) {
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
      numValueBits = typeDeclInfo.getNumFieldsInType(valType);
    }

    // Determine if we should reject mutations after initialization.
    bool isLet = false;
    if (val) {
      if (auto varLet = val.getDefiningOp<VarLetDeclOp>())
        isLet = varLet.getKind() == VarLetDeclKind::Let;
    }

    valueInfoIndex[val] = valueInfos.size();
    valueInfos.push_back({val, firstValueBit, firstValueBit + numValueBits,
                          trackable.startsUninit, trackable.endsUninit,
                          trackable.isIndirect,
                          trackable.isFullObjectLiveOnEntry, isLet,
                          /*hasErrorDiagnosed=*/false,
                          /*isMutatedWhenInitialized=*/false});
  }

  /// Return a reference to the entire value with the specified ID.
  ValueRef getFullValueRef(unsigned valueId) const {
    const auto &entry = valueInfos[valueId];
    return ValueRef{valueId, entry.startValueBit, entry.endValueBit,
                    entry.isIndirect};
  }

  /// Given a pointer or value that is being accessed by an operation, return
  /// the ValueRef for the object being tracked or null if untracked.
  ValueRef getValueRef(Value value) const;

  /// Return the total number of bits we need to track in the bitvector.
  unsigned getNumTotalBits() const {
    return !valueInfos.empty() ? valueInfos.back().endValueBit : 0;
  }

  raw_ostream &printBV(const BitVector &bits, raw_ostream &os) const;
  LLVM_DUMP_METHOD void dumpBV(const BitVector &bits) const {
    auto &os = llvm::errs();
    printBV(bits, os) << "\n";
    os.flush();
  }

  LLVM_DUMP_METHOD void dump() const;
  void printFuncName(raw_ostream &os) const;

  // Get the location of the function we're scanning.
  Location getFuncLocation() { return func.getLoc(); }

private:
  SmallVector<ValueInfo> valueInfos;
  DenseMap<Value, unsigned> valueInfoIndex;
  LIT::FuncOp func;
};
} // namespace

raw_ostream &ValueSet::printBV(const BitVector &bv, raw_ostream &os) const {
  if (bv.size() != getNumTotalBits())
    return os << "WRONG LENGTH BIT VECTOR";

  os << '[';
  llvm::interleave(
      valueInfos,
      [&](const ValueInfo &vi) {
        for (size_t i = vi.startValueBit, e = vi.endValueBit; i != e; ++i)
          os << (bv.test(i) ? '1' : '0');
      },
      [&]() { os << ' '; });
  return os << ']';
}

void ValueSet::printFuncName(raw_ostream &os) const {
  if (auto funcOp = dyn_cast<LIT::FuncOp>(func))
    os << "'" << funcOp.getName() << "'";
  else
    os << "(non func)";
}

void ValueSet::dump() const {
  auto &os = llvm::errs();
  os << "ValueSet with " << valueInfos.size() << " values for ";
  printFuncName(os);
  os << "\n";
  os << "  SI = startsInit, EI = endsInit, [*] = isIndirect";
  os << "  FL=isFullObjectLiveOnEntry, ERR = hadErrorDiag\n";

  for (auto [idx, info] : llvm::enumerate(valueInfos)) {
    os << "  #" << idx << " [" << info.startValueBit << ":" << info.endValueBit
       << ")";

    if (!info.startsUninit)
      os << " SI";
    if (!info.endsUninit)
      os << " EI";
    if (info.isIndirect)
      os << " [*]";
    if (info.isFullObjectLiveOnEntry)
      os << " FL";
    if (info.hasErrorDiagnosed)
      os << " ERR";
    os << "\t";

    if (!info.value) {
      os << "<<null sentinel>>\n";
      continue;
    }

    // If this is a function argument, be nice and include the name.
    if (auto bbArg = dyn_cast<BlockArgument>(info.value)) {
      if (auto fn =
              dyn_cast_or_null<LIT::FuncOp>(bbArg.getOwner()->getParentOp()))
        os << fn.getSignature().getArgName(bbArg.getArgNumber()) << " ";
    }

    os << info.value << "\n";
  }
  os.flush();
}

/// Given a pointer that is being accessed indirectly by an operation, return
/// the value number being referenced, or zero if not tracked.
ValueRef ValueSet::getValueRef(Value value) const {
  // If this is a value we're tracking, return it.
  auto it = valueInfoIndex.find(value);
  if (it != valueInfoIndex.end())
    return getFullValueRef(it->second);

  // If this is a GEP, check the base and focus in on a field of it.
  // TODO(references) remove this.
  if (auto structGEP = value.getDefiningOp<LIT::StructGEPOp>()) {
    ValueRef baseVal = getValueRef(structGEP.getContainer());
    if (!baseVal || !baseVal.isIndirect)
      return {};

    // Figure out what subset of elements we have indexed to.
    auto containerType = structGEP.getContainer().getType().getElementType();
    unsigned fieldOffset = typeDeclInfo.getFieldIndex(
        cast<DeclRefType>(containerType), structGEP.getFieldAttr());
    unsigned startBit = baseVal.startBit + fieldOffset;
    auto resultType = structGEP.getType().getElementType();
    return ValueRef{baseVal.valueId, startBit,
                    startBit + typeDeclInfo.getNumFieldsInType(resultType),
                    /*isIndirect=*/true};
  }

  // If this is a GER, check the base and focus in on a field of it.
  if (auto structGER = value.getDefiningOp<RefStructGEROp>()) {
    ValueRef baseVal = getValueRef(structGER.getContainer());
    if (!baseVal || !baseVal.isIndirect)
      return {};

    // Figure out what subset of elements we have indexed to.
    auto containerType = structGER.getContainer().getType().getElementAsType();
    unsigned fieldOffset = typeDeclInfo.getFieldIndex(
        cast<DeclRefType>(containerType), structGER.getFieldAttr());
    unsigned startBit = baseVal.startBit + fieldOffset;
    auto resultType = structGER.getType().getElementAsType();
    return ValueRef{baseVal.valueId, startBit,
                    startBit + typeDeclInfo.getNumFieldsInType(resultType),
                    /*isIndirect=*/true};
  }

  // If this is a load from a lifetime tracked indirect value, then this is a
  // borrow of that value.
  if (auto load = value.getDefiningOp<POP::LoadOp>())
    if (auto valueRef = getValueRef(load.getPtr())) {
      if (valueRef.isIndirect) {
        // The parser doesn't emit all the lifetime stuff for trivial types,
        // so don't track them either.
        if (typeDeclInfo.isRegisterPassableTrivial(load.getType()))
          return {};

        valueRef.isIndirect = false;
        return valueRef;
      }
    }

  if (auto load = value.getDefiningOp<RefLoadOp>())
    if (auto valueRef = getValueRef(load.getRef())) {
      if (valueRef.isIndirect) {
        // The parser doesn't emit all the lifetime stuff for trivial types,
        // so don't track them either.
        if (typeDeclInfo.isRegisterPassableTrivial(load.getType()))
          return {};

        valueRef.isIndirect = false;
        return valueRef;
      }
    }

  // If this is a RefToPointerOp get the underlying ref.
  if (auto refToPointer = value.getDefiningOp<RefToPointerOp>())
    return getValueRef(refToPointer.getRef());

  // If this is a RebindOp get the underlying ref.
  if (auto rebind = value.getDefiningOp<RebindOp>())
    return getValueRef(rebind.getOperand());

  // Otherwise, we don't know what this is.
  return ValueRef();
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

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkOp(Operation &op);
  ValueRef checkUse(Value value, Operation &op);
  ValueRef checkDef(Value value, Operation &op);
  ValueRef checkConsume(Value value, Operation &op);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is the set of values known to be live at this point.
  BitVector liveValues;

  /// This is the set of values known to be mutated on any prior paths, used for
  /// let-var warnings and errors about lazy-initialized let's.  This can be
  /// true when the value isn't live because the value has since been consumed.
  BitVector everMutatedValues;

  /// When analyzing the body of a loop, this bitset indicates what a 'continue'
  /// should intersect with.
  BitVector *continueSet = nullptr, *continueEverMutatedSet = nullptr;
  /// When analyzing the body of a loop, this bitset indicates what a 'break'
  /// should intersect with.
  BitVector *breakSet = nullptr, *breakEverMutatedSet = nullptr;
  /// When analyzing the body of a try, this bitset indicates what a
  /// 'raise' should intersect with.
  BitVector *raiseSet = nullptr, *raiseEverMutatedSet = nullptr;
};
} // namespace

void UninitializedValueScan::dump() const {
  auto &os = llvm::errs();
  if (valueSet.getValueInfos().size() < 10) {
    valueSet.dump();
    os << "\n";
  }

  os << "UninitializedValueScan for ";
  valueSet.printFuncName(os);
  os << "\n  live = ";
  valueSet.printBV(liveValues, os) << "\n  mutated = ";
  valueSet.printBV(everMutatedValues, os) << "\n";

  if (raiseSet) {
    os << " raise: ";
    valueSet.printBV(*raiseSet, os) << "\n";
  }
  if (breakSet) {
    os << " break: ";
    valueSet.printBV(*breakSet, os) << "\n";
  }
  if (continueSet) {
    os << " continue: ";
    valueSet.printBV(*continueSet, os) << "\n";
  }
  os.flush();
}

static Type digIntoTypeAtFieldOffset(Type type, unsigned firstInvalidOffset,
                                     unsigned nextValidOffset,
                                     InFlightDiagnostic &diag,
                                     TypeDeclInfo &typeDeclInfo) {
  // Dig into the type to get to the right field.
  while (firstInvalidOffset) {
    // If this is the full-object bit for this entire type, then we found the
    // problem.
    if (firstInvalidOffset + 1 == typeDeclInfo.getNumFieldsInType(type))
      return type;

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
static void addBadValueNameToDiag(ValueRef valueRef, const BitVector &bits,
                                  ValueSet &valueSet,
                                  mlir::InFlightDiagnostic &diag) {
  const ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);

  diag << "'" << valueEntry.getName().str();
  // If the whole value is missing, then don't add any field information.
  if (valueSet.getFullValueRef(valueRef.valueId).isAllMissing(bits)) {
    diag << "'";
    return;
  }

  // Figure out what the end of the field bits are so we can report the first
  // fields.  The full object ends with a bit to track whether the whole value
  // is initialized which we don't want to track.
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
  auto type = valueRef.getValueType(valueEntry.value);
  // Emit the field prefix for the specified type.
  digIntoTypeAtFieldOffset(type, firstMissingFieldNo - fullValueStartBit,
                           firstPresentFieldNo - fullValueStartBit, diag,
                           valueSet.typeDeclInfo);
  diag << "'";
}

// Verify that the specified ValueRef is live at this point, diagnosing an
// error at the specified operation if not.
ValueRef UninitializedValueScan::checkUse(Value value, Operation &op) {
  ValueRef valueRef = valueSet.getValueRef(value);
  if (!valueRef)
    return valueRef;

  // If the value is live then all is good.
  if (valueRef.isAllPresent(liveValues))
    return valueRef;

  // Ok, it isn't, gear up to see how to best report the error.
  ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);
  if (valueEntry.hasErrorDiagnosed)
    return valueRef; // Only report one error per symbolic value.
  valueEntry.hasErrorDiagnosed = true;

  // If the fields are all valid except for the whole-object bit, then the user
  // tried to initialize a value by initializing all its fields.  Reject this
  // with a customized error.
  if (valueRef.isIndirect && valueRef.endBit == valueEntry.endValueBit &&
      valueRef.getSubfield(0, valueRef.getNumBits() - 1)
          .isAllPresent(liveValues) &&
      valueRef.getNumBits() != 1) {
    auto diag = mlir::emitError(op.getLoc(), "'")
                << valueEntry.getName().str()
                << "' used with all fields manually initialized "
                   "but without calling an '__init__' method";
    diag.attachNote(valueEntry.value.getLoc())
        << "'" << valueEntry.getName().str() << "' declared here";
    return valueRef;
  }

  // Specialize diagnostics for returns because it can be confusing why they are
  // "using" argument values otherwise.
  auto diag = mlir::emitError(op.getLoc());
  if (isa<KGEN::ReturnOp>(op)) {
    addBadValueNameToDiag(valueRef, liveValues, valueSet, diag);
    diag << " is uninitialized at ";

    // Diagnostics with implicit function returns can be confusing because the
    // Location of the return op is set to the function entry.  Make it
    // explicit when we're complaining about this.
    if (op.getLoc() == valueSet.getFuncLocation())
      diag << "the implicit ";

    diag << "return from this function";
  } else {
    diag << "use of uninitialized value ";

    // If some fields are present and others are missing, complain about the
    // first whole field that is missing.
    addBadValueNameToDiag(valueRef, liveValues, valueSet, diag);
  }
  diag.attachNote(valueEntry.value.getLoc())
      << "'" << valueEntry.getName().str() << "' declared here";

  return valueRef;
}

ValueRef UninitializedValueScan::checkDef(Value value, Operation &op) {
  ValueRef valueRef = valueSet.getValueRef(value);
  if (!valueRef)
    return valueRef;

  // If we are overwriting a value that has already been specified, then the
  // underlying value must be declared a 'var' and not a 'let'.
  if (!valueRef.isAllMissing(everMutatedValues)) {
    ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);

    // If this was declared as a let, then this is an error.
    if (info.isLet && !info.hasErrorDiagnosed) {
      auto diag =
          mlir::emitError(op.getLoc(), "invalid mutation of immutable value ");
      addBadValueNameToDiag(valueRef, everMutatedValues, valueSet, diag);
      diag.attachNote(info.value.getLoc())
          << "'" << info.getName().str() << "' declared here";
      info.hasErrorDiagnosed = true;
    }

    // If this is a var, then just notice the mutation so it doesn't get
    // suggested to promote to a let.
    info.isMutatedWhenInitialized = true;
  }

  // Finally, marks its value live so any use after this isn't treated as
  // uninitialized.
  valueRef.markBits(liveValues, true);
  valueRef.markBits(everMutatedValues, true);
  return valueRef;
}

ValueRef UninitializedValueScan::checkConsume(Value value, Operation &op) {
  ValueRef valueRef = valueSet.getValueRef(value);
  if (!valueRef)
    return valueRef;

  // If tracked, marks its value as dead.
  if (!valueSet.typeDeclInfo.isRegisterPassableTrivial(
          valueRef.getValueType(value)))
    valueRef.markBits(liveValues, false);

  // Mark the value as mutated.
  valueRef.markBits(everMutatedValues, true);
  return valueRef;
}

void UninitializedValueScan::scanFunction(LIT::FuncOp func) {
  // Initialize the BitVector with all the elements that are live-in.  We treat
  // all values live at the start of the function (even before they are actually
  // defined) because we know that all uses must be after them due to SSA
  // dominance.
  liveValues.resize(valueSet.getNumTotalBits());
  everMutatedValues.resize(valueSet.getNumTotalBits());
  for (const ValueInfo &info : valueSet.getValueInfos())
    if (!info.startsUninit) {
      // If the whole value is live on entry, notice that.
      liveValues.set(info.startValueBit, info.endValueBit);
    } else if (info.isFullObjectLiveOnEntry) {
      // If /just/ the full object bit is live on entry, set it.
      liveValues.set(info.endValueBit - 1);
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
  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.
  if (isa<DebugInfo::ValueOp>(op))
    return;

  // This op is handled when used.
  if (isa<LIT::StructGEPOp, RefStructGEROp, RebindOp,
          // TODO(references): remove these.
          RefToPointerOp, mlir::UnrealizedConversionCastOp>(op))
    return;

  // A store of a whole value is an initialization.
  // TODO(references): Remove POP::StoreOp.
  if (isa<LIT::RefStoreOp, POP::StoreOp, OwnershipDefLValueOp>(op)) {
    // Mark the pointer as being mutated.
    checkDef(op.getOperands().back(), op);
    return;
  }

  // A load is a use of whatever fields are being referenced.
  if (isa<POP::LoadOp, RefLoadOp, LoadConsumeOp>(op)) {
    checkUse(op.getOperand(0), op);
    if (isa<LoadConsumeOp>(op))
      checkDef(op.getResult(0), op);
    return;
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (isa<LIT::CallSignatureOp, KGENCallOpInterface>(op)) {
    SignatureType signature;
    ValueRange operands;
    if (auto directCall = dyn_cast<KGENCallOpInterface>(op)) {
      signature = directCall.getCalleeType();
      operands = directCall.getArguments();
    } else {
      auto callSig = cast<LIT::CallSignatureOp>(op);
      signature = callSig.getCallee().getType();
      operands = callSig.getArguments();
    }

    assert(isa<CreateClosureOp>(op) ||
           signature.getInputConventions().size() == operands.size());
    for (auto [convention, operand] :
         llvm::zip(signature.getInputConventions(), operands)) {
      switch (convention) {
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::OwnedInMem:
        checkUse(operand, op); // Live -> dead
        checkConsume(operand, op);
        break;
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::BorrowedInReg:
        checkUse(operand, op); // Live -> live
        break;
      case ValueInputConvention::ByRef:
        checkUse(operand, op); // Life -> Live
        checkDef(operand, op);
        break;
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        checkDef(operand, op); // This call defines the by-ref result.
        break;
      case ValueInputConvention::None:
        llvm_unreachable("none convention not permitted in lit");
      }
    }

    // If the result is defining an owned register value, then we treat this as
    // a definition.
    if (signature.hasOwnedRegisterResult())
      checkDef(op.getResult(0), op);

    return;
  }

  // The lit.ownership.mark_destroyed op consumes the whole object bit of a
  // value only, but not its fields.
  if (auto markDestroyed = dyn_cast<LIT::OwnershipMarkDestroyedOp>(op)) {
    if (auto valueRef = valueSet.getValueRef(markDestroyed.getValue())) {
      valueRef = valueRef.getSubfield(valueRef.getNumBits() - 1, 1);
      // If the consumed bit is live then all is good, otherwise there is an
      // error and it will be diagnosed below.
      if (valueRef.isAllPresent(liveValues))
        return;
    }
  }

  // If this operation has a direct use of a value we are tracking, consider
  // it a use that must be initialized.  This notably includes LoadOp.
  for (Value operand : op.getOperands())
    checkUse(operand, op);

  // lit.letreg.decl defines its own value after using its operand.
  if (isa<LetRegDeclOp>(op)) {
    // Operand use already checked above.
    checkDef(op.getResult(0), op);
    return;
  }

  // lit.ownership.end_lifetime consumes its operand then defines its result.
  if (auto ownershipEnd = dyn_cast<OwnershipEndLifetimeOp>(op)) {
    // Operand use already checked above.
    checkConsume(ownershipEnd.getOperand(), op);
    checkDef(ownershipEnd.getResult(), op);
    return;
  }

  // OwnershipMakePointerLValue is a def if liveOnEntry.
  if (auto makePointer = dyn_cast<OwnershipMakePointerLValue>(op)) {
    // Operand use already checked above.
    checkDef(makePointer.getOperand(), op);
    if (makePointer.getLiveOnEntry())
      checkDef(makePointer.getResult(), op);
  }

  // If this is a kgen.return then we have an exit from the function
  // (including early returns and exception raises that leave the function).
  // Check that *all* of the values we are tracking are managed correctly.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op)) {
    for (const ValueInfo &valueInfo :
         llvm::drop_begin(valueSet.getValueInfos())) {
      // If the value doesn't need to be live at end of function, ignore it.
      if (valueInfo.endsUninit)
        continue;

      // If this is a `isFullObjectLiveOnEntry` value (i.e., the 'self' member
      // in an __init__) then it is actually not used on the error path, only
      // the normal path.
      if (valueInfo.isFullObjectLiveOnEntry && isa<LIT::ErrorReturnOp>(op))
        continue;

      // Otherwise, it must be live at return/raise.
      checkUse(valueInfo.value, op);
    }

    // Indicate that all values are live after the return so that an early
    // return in an 'if' will get properly intersected with the other side of
    // the branch.
    liveValues.set();
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
    *breakEverMutatedSet |= everMutatedValues;
    return;
  }
  if (isa<HLCF::ContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    *continueSet &= liveValues;
    *continueEverMutatedSet |= everMutatedValues;
    return;
  }
  if (isa<LIT::TryRaiseOp>(op)) {
    assert(raiseSet && "Not in a 'try'?");
    *raiseSet &= liveValues;
    *raiseEverMutatedSet |= everMutatedValues;
    return;
  }

  // 'if' operations treat the condition as a use but have live outs that are
  // the intersection of the live values produced by the then/else branches.
  if (isa<HLCF::IfOp, ParamIfOp, HandleVariantOp>(op)) {
    assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
           op.getRegion(1).hasOneBlock() &&
           "if-like op should have two single-block regions");
    BitVector liveValuesCopy = liveValues;
    BitVector everMutatedCopy = everMutatedValues;
    scanBlock(op.getRegion(0).front());
    liveValuesCopy.swap(liveValues);
    everMutatedCopy.swap(everMutatedValues);
    scanBlock(op.getRegion(1).front());
    liveValues &= liveValuesCopy;
    everMutatedValues |= everMutatedCopy;

    // HandleVariant defines an owned value as its result, it is produced by an
    // enclosing lit.yield.
    if (isa<HandleVariantOp>(op))
      checkDef(op.getResult(0), op);
    return;
  }

  // For a loop, we analyze the body of the loop with the known live-ins but
  // capture a new sets for continue and break results.
  if (auto loopOp = dyn_cast<HLCF::LoopOp>(op)) {
    UninitializedValueScan bodySets(valueSet);
    // Loops are transparent to raise.
    bodySets.raiseSet = raiseSet;
    bodySets.raiseEverMutatedSet = raiseEverMutatedSet;

    // The default continueSet is the live-in set of values.  This can lose
    // values if some 'continue' path through the body of the loop consumes a
    // value.
    BitVector continueSet(liveValues);
    bodySets.continueSet = &continueSet;
    BitVector continueEverMutatedSet(everMutatedValues);
    bodySets.continueEverMutatedSet = &continueEverMutatedSet;

    // The 'breakSet' of the loop body will be the live outs of the loop.  We
    // use the existing liveValues that we'll continue with for that set, but
    // need to start it out thinking that everything is live so intersections
    // from the body work correctly.
    liveValues.set();
    bodySets.breakSet = &liveValues;
    bodySets.breakEverMutatedSet = &everMutatedValues;

    // Iteratively scan the loop body until the live-in set converges.  This is
    // a trivial lattice with each bit converging to "not live in", so we know
    // this will terminate.
    size_t numLiveIn;
    do {
      numLiveIn = continueSet.count();
      // Scan the body: any breaks will intersect their live-out set with
      // 'breakSet', and any continues will intersect their live-out set with
      // 'continueSet'.
      bodySets.liveValues = continueSet;
      bodySets.everMutatedValues = continueEverMutatedSet;
      bodySets.scanBlock(loopOp.getBody().front());

      // If any bits got cleared from the continueSet then we need to iterate.
    } while (continueSet.count() != numLiveIn);
    // Any code after the loop continues on with the breaks valid.
    return;
  }

  if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
    UninitializedValueScan bodySets(valueSet);
    // Our current live-in set is live-in to the try body.
    bodySets.liveValues = liveValues;
    bodySets.everMutatedValues = everMutatedValues;

    // Try is transparent to break/continue.
    bodySets.continueSet = continueSet;
    bodySets.continueEverMutatedSet = continueEverMutatedSet;
    bodySets.breakSet = breakSet;
    bodySets.breakEverMutatedSet = breakEverMutatedSet;

    // We capture all the common values live-out of raise's as being the live-in
    // to the except block.
    BitVector exceptSet(liveValues.size(), true);
    bodySets.raiseSet = &exceptSet;
    BitVector exceptEverMutatedSet(liveValues.size(), false);
    bodySets.raiseEverMutatedSet = &exceptEverMutatedSet;
    bodySets.scanBlock(tryOp.getTryRegion().front());

    // The live-ins to the except block are the exceptSet.
    for (Value arg : tryOp.getExceptRegion().getArguments())
      if (ValueRef ref = valueSet.getValueRef(arg))
        ref.markBits(exceptSet, true);
    liveValues = std::move(exceptSet);
    everMutatedValues = std::move(exceptEverMutatedSet);
    scanBlock(tryOp.getExceptRegion().front());

    // The live-out set of the bodySet is the live-in to the else block, but
    // exceptions raised in it go out of the try.
    bodySets.raiseSet = raiseSet;
    bodySets.raiseEverMutatedSet = raiseEverMutatedSet;
    bodySets.scanBlock(tryOp.getElseRegion().front());

    // The fall through live values are the intersection from the except and
    // else blocks.
    liveValues &= bodySets.liveValues;
    everMutatedValues |= bodySets.everMutatedValues;
    return;
  }
}

//===----------------------------------------------------------------------===//
// DestructorInsertion
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the third pass over a function body, which
/// inserts destructors after the last use of values.
struct DestructorInsertion {
  DestructorInsertion(ValueSet &valueSet, SmallVector<Operation *> &opsToRemove)
      : valueSet(valueSet), opsToRemove(opsToRemove) {}
  DestructorInsertion(const DestructorInsertion &existing) = delete;
  DestructorInsertion(DestructorInsertion &&existing) = default;

  static DestructorInsertion copy(const DestructorInsertion &existing) {
    DestructorInsertion result(existing.valueSet, existing.opsToRemove);
    result.consumedValues = existing.consumedValues;
    result.raiseSet = existing.raiseSet;
    result.breakSet = existing.breakSet;
    result.continueSet = existing.continueSet;
    result.dryRun = existing.dryRun;
    result.functionSignature = existing.functionSignature;
    return result;
  }

  void scanFunction(LIT::FuncOp func);
  void scanBlock(Block &body);

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkOp(Operation &op);
  void markConsumed(Value value, Operation &op);
  void checkUse(Value value, Operation &op);
  void checkUse(Value value, mlir::ImplicitLocOpBuilder &builder,
                Operation *opWithUse);
  void checkDef(Value value, Operation &op);
  void destroyValuesAtEntry(const BitVector &entries, Block &block,
                            Location loc);
  void destroyValueIfNeeded(Value value, ValueRef valueRef,
                            mlir::ImplicitLocOpBuilder &builder,
                            Operation *opWithUse);

  LogicalResult elideCopyDestroyPair(Value value, Type destroyedType,
                                     Operation *opWithUse);
  void emitDestructorCallAt(Value value, ValueRef valueRef,
                            mlir::ImplicitLocOpBuilder &builder,
                            Operation *opWithUse);

  void checkIfLikeOp(Operation &operation,
                     BitVector &expectedConsumptionInThenNotElse);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is a set of operations that are removed after destructor processing
  /// has completed.  This is used to elide copy ctors.
  SmallVector<Operation *> &opsToRemove;

  /// This is the set of values known to be used below this point, so they
  /// should not be destroyed if there are uses.  Any use of a value /not/ in
  /// this set will be a last use that does get destroyed.
  BitVector consumedValues;

  /// When true, scanning an operation or block will not insert destructors, and
  /// certain invariants don't hold.  This is used when processing loops,
  /// because we need to iterate to a fixed point of values live in from
  /// continue blocks before inserting destructors.
  bool dryRun = false;

  /// When analyzing the body of a try, this bitset indicates what a 'raise'
  /// should produce based on its surrounding 'try's except block's expectation.
  BitVector *raiseSet = nullptr;

  /// When analyzing the body of a loop, these bitset indicates what a 'break'
  /// or 'continue' should produce based on its consumed value set for the
  /// surrounding loop.
  BitVector *breakSet = nullptr;
  BitVector *continueSet = nullptr;

  /// This is the signature of the current function being analyzed.
  SignatureType functionSignature;
};
} // namespace

void DestructorInsertion::checkIfLikeOp(
    Operation &ifElseOp, BitVector &expectedConsumptionInThenNotElse) {
  assert(ifElseOp.getNumRegions() == 2 && ifElseOp.getRegion(0).hasOneBlock() &&
         ifElseOp.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");
  BitVector thenConsumedValues = consumedValues;
  scanBlock(ifElseOp.getRegion(0).front());

  // Scan 'else' block.
  thenConsumedValues.swap(consumedValues);
  scanBlock(ifElseOp.getRegion(1).front());
  // At this point, 'thenConsumedValues' is the set of upwardly consumed
  // values from the 'then' block and 'consumedValues' is the set of upwardly
  // consumed values from the else branch.  See if they disagree.
  BitVector disagreements = consumedValues;
  disagreements ^= thenConsumedValues;
  // If they agree, then we're done if not, we'll have to destroy fields to
  // make them agree.
  if (disagreements.none())
    return;

  // If we are in a dryrun, just compute the union of the two sets.
  if (dryRun) {
    consumedValues |= disagreements;
    return;
  }

  // Otherwise we have to emit destructors to get the branches to line up.
  // If the true branch consumed values that the false branch didn't, then
  // we need to destroy those corresponding values in the false branch.
  BitVector consumedInElseButNotThen = consumedValues;
  consumedInElseButNotThen &= disagreements;
  destroyValuesAtEntry(consumedInElseButNotThen, ifElseOp.getRegion(0).front(),
                       ifElseOp.getLoc());

  BitVector consumedInThenButNotElse = thenConsumedValues;
  consumedInThenButNotElse &= disagreements;
  BitVector &inverse = expectedConsumptionInThenNotElse.flip();
  consumedInThenButNotElse &= inverse;
  destroyValuesAtEntry(consumedInThenButNotElse, ifElseOp.getRegion(1).front(),
                       ifElseOp.getLoc());

  // Restore consumedValues to the merged set.
  consumedValues = thenConsumedValues;
  consumedValues |= consumedInElseButNotThen;
}

void DestructorInsertion::dump() const {
  auto &os = llvm::errs();
  if (valueSet.getValueInfos().size() < 10) {
    valueSet.dump();
    os << "\n";
  }

  os << "DestructorInsertion for ";
  valueSet.printFuncName(os);
  if (dryRun)
    os << " [DRYRUN]";
  os << "\n  ";
  valueSet.printBV(consumedValues, os) << "\n";

  if (raiseSet) {
    os << " raise: ";
    valueSet.printBV(*raiseSet, os) << "\n";
  }
  if (breakSet) {
    os << " break: ";
    valueSet.printBV(*breakSet, os) << "\n";
  }
  if (continueSet) {
    os << " continue: ";
    valueSet.printBV(*continueSet, os) << "\n";
  }
  os.flush();
}

void DestructorInsertion::scanFunction(LIT::FuncOp func) {
  if (auto fnInterface = dyn_cast<FuncInterface>(func.getOperation()))
    functionSignature = fnInterface.getSignature();
  else // Unknown function kind.
    return;

  consumedValues.resize(valueSet.getNumTotalBits());
  consumedValues.set(0); // Never destroy slot 0, it is already destroyed.

  // Scan the body of the function.
  Block &funcBody = func.getFunctionBody().front();
  scanBlock(funcBody);

  for (auto [valueID, valueInfo] : llvm::enumerate(valueSet.getValueInfos())) {
    if (valueInfo.startsUninit || valueID == 0)
      continue;

    // If an op result initialized on entry was overwritten, make sure to
    // destroy the value.
    if (auto result = dyn_cast<OpResult>(valueInfo.value)) {
      Operation *op = result.getOwner();
      mlir::ImplicitLocOpBuilder builder(result.getLoc(), op->getBlock(),
                                         ++op->getIterator());
      checkUse(valueInfo.value, builder, /*opWithUse=*/nullptr);
      continue;
    }

    // If any owned argument values are unconsumed then they must be unused.
    // Emit their destructor calls at the start of the function by acting as
    // though there is a use.
    Location loc = valueInfo.value.getLoc();
    if (DebugInfo::DISubprogramAttr scope =
            DebugInfo::extractScope(cast<mlir::FunctionOpInterface>(*func)))
      loc = FusedLoc::get(loc.getContext(), {loc}, scope);

    mlir::ImplicitLocOpBuilder builder(loc, &funcBody, funcBody.begin());
    checkUse(valueInfo.value, builder, /*opWithUse=*/nullptr);
  }
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
  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.
  if (isa<DebugInfo::ValueOp>(op))
    return;

  // This op is handled when used.
  if (isa<LIT::StructGEPOp, RefStructGEROp, RebindOp,
          // TODO(references): remove these.
          RefToPointerOp, mlir::UnrealizedConversionCastOp>(op))
    return;

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (isa<LIT::CallSignatureOp, KGENCallOpInterface>(op)) {
    SignatureType signature;
    if (auto directCall = dyn_cast<KGENCallOpInterface>(op))
      signature = directCall.getCalleeType();
    else
      signature = cast<LIT::CallSignatureOp>(op).getCallee().getType();
    ValueRange operands;
    if (isa<LIT::CallSignatureOp>(op))
      operands = cast<LIT::CallSignatureOp>(op).getArguments();
    else
      operands = op.getOperands();

    // If the result is defining an owned register value, treat it as a def.
    if (signature.hasOwnedRegisterResult())
      checkDef(op.getResult(0), op);

    assert(isa<CreateClosureOp>(op) ||
           signature.getInputConventions().size() == operands.size());
    for (auto [convention, operand] :
         llvm::zip(signature.getInputConventions(), operands)) {
      switch (convention) {
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::OwnedInMem:
        // This consumes the value, so it isn't dead going upwards.
        valueSet.getValueRef(operand).markBits(consumedValues, true);
        break;
      case ValueInputConvention::BorrowedInReg:
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::ByRef:
        checkUse(operand, op);
        break;
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        // This defines the memory it writes to.
        checkDef(operand, op);
        break;
      case ValueInputConvention::None:
        llvm_unreachable("none convention not permitted in lit");
      }
    }
    return;
  }

  // LetReg takes ownership of its operand value and defines its own value.
  if (auto letReg = dyn_cast<LetRegDeclOp>(op)) {
    // This defines the result value.  Emit a destructor if unused.
    checkDef(letReg, op);
    // This consumes its input.
    markConsumed(letReg.getOperand(), op);
    return;
  }

  // A store consumes a value and overwrites the destination.
  // TODO(references): Remove POP::StoreOp.
  if (auto storeOp = dyn_cast<POP::StoreOp>(op)) {
    markConsumed(storeOp.getArg(), op);
    checkDef(storeOp.getPtr(), op);
    return;
  }

  // A store consumes a value and overwrites the destination.
  if (auto storeOp = dyn_cast<LIT::RefStoreOp>(op)) {
    markConsumed(storeOp.getArg(), op);
    checkDef(storeOp.getRef(), op);
    return;
  }

  // A load is a use of whatever fields are being referenced.  If this is the
  // /last/ use of a value, emit a destructor of that value.  LoadOps are used
  // to model a /borrow/ of the underlying value, so they don't define a new
  // value.
  if (auto loadOp = dyn_cast<POP::LoadOp>(op)) {
    checkUse(loadOp.getPtr(), op);
    return;
  }
  if (auto loadOp = dyn_cast<RefLoadOp>(op)) {
    checkUse(loadOp.getRef(), op);
    return;
  }

  // These operations consume their operands and define a result.
  if (isa<LoadConsumeOp, OwnershipEndLifetimeOp, LIT::StructCreateOp>(op)) {
    checkDef(op.getResult(0), op);
    for (auto operand : op.getOperands())
      markConsumed(operand, op);
    return;
  }

  // OwnershipMakePointerLValue is a def if liveOnEntry.
  if (auto makePointer = dyn_cast<OwnershipMakePointerLValue>(op)) {
    checkUse(makePointer.getOperand(), op);
    if (makePointer.getLiveOnEntry())
      checkUse(makePointer.getResult(), op);
  }

  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp>(op)) {
    consumedValues.reset();
    consumedValues.set(0); // Never destroy slot 0, it is already destroyed.
    for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
      if (!valueInfo.endsUninit)
        consumedValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
    }

    // If the result operand is ownedresult, then consume it.
    if (functionSignature.hasOwnedRegisterResult())
      markConsumed(op.getOperand(0), op);
    return;
  }

  // A yield from a HandleVariantOp consumes the operand.
  if (isa<YieldOp>(op)) {
    markConsumed(op.getOperand(0), op);
    return;
  }

  if (auto errorReturn = dyn_cast<ErrorReturnOp>(op)) {
    // This marks a point where we abandoned an effort to initialize
    // a value fully. Call the destructor on the members that we
    // initialized.
    for (const ValueInfo &valueInfo :
         llvm::drop_begin(valueSet.getValueInfos()))
      // If this value is supposed to be initialized upon function exit, check
      // for partial initialization.
      if (!valueInfo.endsUninit && valueInfo.startsUninit) {
        Value topLevelValue = valueInfo.value;
        ValueRef topLevelValueRef = valueSet.getValueRef(valueInfo.value);
        BitVector original = consumedValues;
        // Pretend for a moment that the top level value has not been consumed.
        // Otherwise, we cannot generate destructor calls.
        topLevelValueRef.markBits(consumedValues, false);

        /// Perform destruction if needed.
        DeclRefType valueType =
            dyn_cast<DeclRefType>(topLevelValueRef.getValueType(topLevelValue));
        if (!valueType)
          continue;

        mlir::ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(),
                                           op.getIterator());
        if (topLevelValueRef.isAllPresent(original)) {
          // If all the values are "consumed" then we haven't hit a field init
          // yet, which means the entire object can be destroyed.
          destroyValueIfNeeded(topLevelValue, topLevelValueRef, builder, &op);
          break;
        }

        // No fields have been initialized yet, so there is nothing to
        // destroy.
        if (topLevelValueRef.isAllMissing(original))
          break;

        // At this point we have some values that have been initialized
        // and some that have not. Let's destroy those that have been
        // initialized.
        LIT::StructDeclOp structDecl =
            valueSet.typeDeclInfo.getStructDeclForType(valueType);

        unsigned offset = 0;
        for (StructFieldOp field : structDecl.getFieldDecls()) {
          unsigned numBits =
              valueSet.typeDeclInfo.getNumFieldsInType(field.getType());
          ValueRef fieldValueRef =
              topLevelValueRef.getSubfield(offset, numBits);
          offset += numBits;
          // Trivial types do not need to be destroyed.
          if (valueSet.typeDeclInfo.isRegisterPassableTrivial(field.getType()))
            continue;
          if (fieldValueRef.isIndirect &&
              original.test(fieldValueRef.startBit)) {
            destroyValueIfNeeded(builder.create<LIT::StructGEPOp>(
                                     op.getLoc(), topLevelValue, field),
                                 fieldValueRef, builder, &op);
          }
        }

        // At this point we have destroyed everything we had initialized. Revert
        // back to the original consume set. We do not need to update the
        // consumed value set because we emitted destructors for subfields,
        // which are not tracked in the ValueSet.
        consumedValues = original;
      } else if (!valueInfo.endsUninit) {
        consumedValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
      }

    // Handle the operand of the return op.
    auto createVariant =
        cast<VariantCreateOp>(errorReturn.getVariant().getDefiningOp());
    auto error = createVariant.getOperand();
    markConsumed(error, op);
    return;
  }
  // A unreachable consumes nothing.  Nothing needs to be destroyed here!
  if (isa<UnreachableOp>(op)) {
    consumedValues.reset();
    consumedValues.set(0); // Never destroy slot 0, it is already destroyed.
    return;
  }

  // A raise will use the consume set that was seen on entry to the enclosing
  // except block.
  if (isa<LIT::TryRaiseOp>(op)) {
    assert(raiseSet && "Not in a 'try'?");
    consumedValues = *raiseSet;
    return;
  }

  if (isa<HLCF::BreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    consumedValues = *breakSet;
    return;
  }
  if (isa<HLCF::ContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    consumedValues = *continueSet;
    return;
  }

  // The lit.ownership.mark_destroyed op consumes the whole object bit of a
  // value only, but not its fields.    This ensures the sub-fields are
  // destroyed but the full object is not.  It is used in destructors primarily.
  if (auto markDestroyed = dyn_cast<LIT::OwnershipMarkDestroyedOp>(op)) {
    if (auto valueRef = valueSet.getValueRef(markDestroyed.getValue()))
      consumedValues.set(valueRef.endBit - 1);
    return;
  }

  if (auto handleVariantOp = dyn_cast<HandleVariantOp>(op)) {
    // The result of HandleVariantOp is always a definition of an owned value
    // that is produced by the enclosed lit.yield operation.
    checkDef(op.getResult(0), op);

    BitVector expectedConsumptionInThenNotElse(consumedValues.size());
    for (Value maybeInitValue : handleVariantOp.getMaybeInitializedValues()) {
      // For each of the initialized values, is this the last reference?
      // If so, generate a destructor call after this op.
      checkUse(maybeInitValue, op);

      // To prevent destructor calls from being generated for uninitialized
      // values in the else block, we mark the exempt values in an expectation
      // bit vector.
      ValueRef uninitValueRef = valueSet.getValueRef(maybeInitValue);
      uninitValueRef.markBits(expectedConsumptionInThenNotElse, true);
    }
    checkIfLikeOp(op, expectedConsumptionInThenNotElse);
    return;
  }

  // 'if' operations propagate the consume sets into each branch, and use the
  // resulting consume sets to make sure the upward propagated set of consumed
  // values is consistent.
  if (isa<HLCF::IfOp, ParamIfOp>(op)) {
    assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
           op.getRegion(1).hasOneBlock() &&
           "if-like op should have two single-block regions");
    BitVector expectedConsumptionInThenNotElse(consumedValues.size());
    checkIfLikeOp(op, expectedConsumptionInThenNotElse);
    return;
  }

  if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
    // The except block is processed with a copy of the consumed value set from
    // the bottom of the try.  After processing it, we know what the consumed
    // values are for the exception block.
    auto exceptSets = DestructorInsertion::copy(*this);
    exceptSets.raiseSet = raiseSet;
    exceptSets.scanBlock(tryOp.getExceptRegion().front());
    // The except block initializes its block arguments, so if these are tracked
    // we must mark them as consumed.
    for (Value blockArg : tryOp.getExceptRegion().getArguments())
      if (ValueRef valueRef = valueSet.getValueRef(blockArg)) {
        if (!exceptSets.consumedValues[valueRef.startBit]) {
          // There were no references to the owned arguments, so generate a
          // destructor at beginning of the block.
          mlir::ImplicitLocOpBuilder builder =
              ImplicitLocOpBuilder::atBlockBegin(
                  tryOp.getExceptRegion().getLoc(),
                  &tryOp.getExceptRegion().front());
          destroyValueIfNeeded(blockArg, valueRef, builder,
                               /*opWithUse=*/nullptr);
          valueRef.markBits(consumedValues, false);
        } else {
          valueRef.markBits(exceptSets.consumedValues, false);
        }
      }

    // The normal flow finishes with the else block, process it to see what the
    // input consumedValues set to the else block is.
    scanBlock(tryOp.getElseRegion().front());

    // Ok, finally we process the try body.  Any 'raise's within the try body
    // use the consumed values set on entry to the except block.
    llvm::SaveAndRestore x(raiseSet, &exceptSets.consumedValues);
    scanBlock(tryOp.getTryRegion().front());
    return;
  }

  // For a loop, we know the consume sets for any break statements, but need to
  // iterate the loop to find the right continue sets to use.
  if (auto loopOp = dyn_cast<HLCF::LoopOp>(op)) {
    auto loopBodySets = DestructorInsertion::copy(*this);
    // Any 'break's within the loop will produce the consume set for the
    // statement immediately after the loop.
    loopBodySets.breakSet = &consumedValues;

    // We start the continueSet with no values set to be consumed.
    BitVector continueSet(consumedValues.size());
    continueSet.set(0); // Never destroy slot 0, it is already destroyed.
    loopBodySets.continueSet = &continueSet;

    // We need to dry run the body evaluation until we get to a stable continue
    // set.
    loopBodySets.dryRun = true;

    // Iteratively scan the loop body until the continue set converges.
    [[maybe_unused]] unsigned numIters = 0;
    while (true) {
      // Scan the body: any breaks will intersect their live-out set with
      // 'breakSet', and any continues will intersect their live-out set with
      // 'continueSet'.
      loopBodySets.scanBlock(loopOp.getBody().front());

      // If the continue set is unchanged, then we converged.
      if (loopBodySets.consumedValues == continueSet)
        break;
      // Otherwise, use the set of values consumed on loop entry as the new
      // continue set.
      std::swap(loopBodySets.consumedValues, continueSet);

      // This should converge trivially as we are setting bits in the continue
      // set, but when we get a consume operator in the future this may be
      // tricky.  Don't fall into an infinite loop on accident.
      ++numIters;
      assert(numIters < 5 && "Loop should converge in a couple iterations");
    }

    // Once we've converged to the right continue set, we can replay one final
    // iteration in execute mode (if the enclosing context is not dryRun mode)
    // to insert destructors.
    if (!dryRun) {
      loopBodySets.dryRun = false;
      loopBodySets.scanBlock(loopOp.getBody().front());
    }
    consumedValues = std::move(loopBodySets.consumedValues);
    return;
  }

  // Otherwise this some other operation that using a SSA value.  If this is the
  // last use of the value, make sure we destroy it when done.
  for (Value operand : op.getOperands())
    checkUse(operand, op);
}

// When the specified value is consumed by an operation we know it doesn't need
// to be destroyed above this point.
void DestructorInsertion::markConsumed(Value value, Operation &op) {
  ValueRef valueRef = valueSet.getValueRef(value);
  if (!valueRef)
    return;

  // If this operation is consuming a sub-element of a value that is already
  // marked to be consumed, then it is being used down below.
  //
  // This happens on code like this, for example:
  //   var a = Pair()
  //   _ = a.x^
  //   use(a.x)
  if (!valueRef.isAllMissing(consumedValues)) {
    // Trivial types don't have __copyinit__ methods, and therefore cannot have
    // ownership tracked for them.
    if (valueSet.typeDeclInfo.isRegisterPassableTrivial(
            valueRef.getValueType(value)))
      return;

    ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
    if (info.hasErrorDiagnosed)
      return;
    ValueRef fullValueRef = valueSet.getFullValueRef(valueRef.valueId);

    auto diag = mlir::emitError(op.getLoc(), "value ");
    // Use a clear bitvector of the right size so we print the entire value
    // being referenced even if only part of it is missing.
    BitVector allMissing(consumedValues.size(), true);
    valueRef.markBits(allMissing, false);
    addBadValueNameToDiag(valueRef, allMissing, valueSet, diag);
    diag << " cannot be consumed, because ";

    if (valueRef.isAllPresent(consumedValues) &&
        (valueRef == fullValueRef ||
         !fullValueRef.isAllPresent(consumedValues))) {
      diag << "it";
    } else {
      // If some fields are present and others are missing, complain about the
      // first whole field that is missing.
      auto aliveValues = consumedValues;
      aliveValues.flip();
      addBadValueNameToDiag(valueRef, aliveValues, valueSet, diag);
    }
    diag << " is used later";
    info.hasErrorDiagnosed = true;
  }

  valueRef.markBits(consumedValues, true);
}

/// This operation uses whatever fields are being referenced.  Iff this is the
/// /last/ use of a value, emit a destructor of the overall value.
void DestructorInsertion::checkUse(Value value, Operation &op) {
  // If needed, emit the destructor immediately after the specified operation.
  auto insertPt = std::next(Block::iterator(&op));
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(), insertPt);
  checkUse(value, builder, /*opWithUse=*/&op);
}

/// Check a use of a value.  Iff this is the /last/ use of the value, emit a
/// destructor of the overall value.  The 'opWithUse' value (if present)
/// indicates the operation performing the use.  This enables copy ctor elision,
/// but this is null at the start of block/function for example.
void DestructorInsertion::checkUse(Value value,
                                   mlir::ImplicitLocOpBuilder &builder,
                                   Operation *opWithUse) {
  ValueRef valueRef = valueSet.getValueRef(value);
  if (!valueRef)
    return;

  ValueInfo &valueInfo = valueSet.getValueInfos()[valueRef.valueId];
  if (valueInfo.hasErrorDiagnosed)
    return;

  // If this is the last use of some value that needs to be destroyed when
  // dead, emit the whole object destructor for the overall value.
  //
  //   init(&aggregate)
  //   use(aggregate.field1)
  //   use(aggregate.field2)  <<-- We are here.
  //
  // Here we emit `dtor(&aggregate)` to destroy the overall value, which will
  // also handle deleting the field in question.
  if (value != valueInfo.value && !consumedValues[valueInfo.endValueBit - 1]) {
    value = valueInfo.value;
    valueRef = valueSet.getFullValueRef(valueRef.valueId);
  }

  // Otherwise, it is possible that that ValueRef is live but the overall object
  // will be consumed, this happens in scenarios like:
  //
  //   init(&aggregate)
  //   use(&aggregate.field1)  <<-- We are here.
  //   ... field1 is not consumed here...
  //   aggregate.field1 = newValue  // overwrite field1.
  //   consume(&aggregate)
  //
  // In this case, we need to destroy field1 after this use.
  destroyValueIfNeeded(value, valueRef, builder, /*opWithUse=*/opWithUse);
}

/// This operation defines the specified value.  If the value is dead on
/// arrival, emit a destructor of the value.
void DestructorInsertion::checkDef(Value value, Operation &op) {
  // If there is no use of the value we are defining, emit a dtor after the op.
  // This happens when we have things like:
  //
  //   init(&aggregate)
  //   ...
  //   aggregate.field1 = newValue  <<-- we are here
  checkUse(value, op);

  // This call defines the result, so anything above it is either dead or
  // needs a destructor if live.
  valueSet.getValueRef(value).markBits(consumedValues, false);
}

/// Recursive version of destroyValueIfNeeded invoked when we know that we are
/// inserting destructors.
void DestructorInsertion::destroyValueIfNeeded(Value value, ValueRef valueRef,
                                               ImplicitLocOpBuilder &builder,
                                               Operation *opWithUse) {
  assert(valueRef && "Only works on valid refs");

  // If we are just computing the consumedValue set, don't actually insert any
  // destructor calls.
  if (dryRun) {
    valueRef.markBits(consumedValues, true);
    return;
  }

  // If nothing in this value needs destroying, then ignore the request.
  if (valueRef.isAllPresent(consumedValues))
    return;

  // Get the type for the value so we can poke at it.
  Type type = valueRef.getValueType(value);

  // If this is a generic type, then emit a generic destructor call. The
  // language guarantees that a destructor is generic for every generic type.
  if (auto generic = dyn_cast<ParamRefType>(type)) {
    if (auto trait = dyn_cast<TraitType>(generic.getParam().getType())) {
      emitDestructorCallAt(value, valueRef, builder, opWithUse);
      valueRef.markBits(consumedValues, true);
      return;
    }
  }

  // If trivial, then we don't have any work to do.
  auto valueType = dyn_cast<DeclRefType>(type);
  if (!valueType) {
    valueRef.markBits(consumedValues, true);
    return;
  }

  // If the entire value needs to be destroyed, then emit a destructor for the
  // whole value.
  if (!consumedValues.test(valueRef.endBit - 1)) {
    // Trivial types don't have __del__ methods.
    if (valueSet.typeDeclInfo.isRegisterPassableTrivial(valueType)) {
      valueRef.markBits(consumedValues, true);
      return;
    }

    // If a field of a value we must destroy is already destroyed, then we have
    // an error, because we cannot run the destructor on the whole object if one
    // of the fields is missing.
    if (!valueRef.isAllMissing(consumedValues)) {
      // Be careful about trivial fields: they don't have correctly tracked
      // lifetimes, and should never be reported as the error for why a value
      // is early destructed.
      unsigned nextBit = 0;
      for (auto field : valueSet.typeDeclInfo.getStructDeclForType(valueType)
                            .getFieldDecls()) {
        unsigned numBits =
            valueSet.typeDeclInfo.getNumFieldsInType(field.getType());
        // If this field has consumed bits, and if has trivial type, force it
        // back to being non-consumed.  This can allow the proper correctness
        // check to work and make the error diagnostic more accurate.
        ValueRef subFieldBits = valueRef.getSubfield(nextBit, numBits);
        if (!subFieldBits.isAllMissing(consumedValues) &&
            valueSet.typeDeclInfo.isRegisterPassableTrivial(
                field.getReboundType(valueType)))
          subFieldBits.markBits(consumedValues, false);
        nextBit += numBits;
      }

      if (!valueRef.isAllMissing(consumedValues)) {
        ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);
        if (valueEntry.hasErrorDiagnosed)
          return; // Only report one error per symbolic value.
        valueEntry.hasErrorDiagnosed = true;

        auto diag = mlir::emitError(builder.getLoc(), "field ");
        auto aliveValues = consumedValues;
        aliveValues.flip();
        // If some fields are present and others are missing, complain about the
        // first whole field that is missing.
        addBadValueNameToDiag(valueRef, aliveValues, valueSet, diag);
        diag << " destroyed out of the middle of a value, preventing the "
                "overall "
                "value from being destroyed";
        valueRef.markBits(consumedValues, false);
      }
    }

    // Ok, everything looks good - actually emit the dtor call here.
    emitDestructorCallAt(value, valueRef, builder, opWithUse);
    valueRef.markBits(consumedValues, true);
    return;
  }

  // Otherwise, we must have an indirect value where some fields are present and
  // some are missing.  Recursively walk the type and destroy just the fields
  // that are missing.
  LIT::StructDeclOp structDecl =
      valueSet.typeDeclInfo.getStructDeclForType(valueType);

  // Convert lit.ref into pointers until
  // TODO(references): pass references to destructors, not pointeres.
  if (isa<RefType>(value.getType()))
    value = builder.create<RefToPointerOp>(value);

  unsigned nextBit = 0;
  for (auto field : structDecl.getFieldDecls()) {
    Operation *fieldVal;
    if (valueRef.isIndirect)
      fieldVal = builder.create<LIT::StructGEPOp>(value, field);
    else
      fieldVal = builder.create<LIT::StructExtractOp>(value, field);

    unsigned numBits =
        valueSet.typeDeclInfo.getNumFieldsInType(field.getType());
    destroyValueIfNeeded(fieldVal->getResult(0),
                         valueRef.getSubfield(nextBit, numBits), builder,
                         /*opWithUse=*/nullptr);

    // If there was no destructor generated (because the element has no
    // destructor) then remove the unused pointer access.
    if (fieldVal->use_empty())
      fieldVal->erase();
    nextBit += numBits;
  }
  // The whole object bit should exist after all the fields.
  assert(valueRef.startBit + nextBit + 1 == valueRef.endBit &&
         "Lost track of bits");
}

/// Return true if the specified 'p1' pointer could point at object or a
/// subcomponent of 'p2'.  This should return true conservatively.
// TODO: In the presence of returned references / lifetimes, we will
// need to be more careful here.
static bool mightPointTo(Value p1, Value p2) {
  assert(isa<PointerType>(p2.getType()));
  // If the value is an integer or other random thing, then it can't point to
  // anything.
  if (!isa<PointerType>(p1.getType()))
    return false;

  Value underlyingP1 = LifetimeTrackable::findUnderlyingValueFromField(p1);
  Value underlyingP2 = LifetimeTrackable::findUnderlyingValueFromField(p2);
  return !underlyingP1 || !underlyingP2 || underlyingP1 == underlyingP2;
}

// Check to see if we can eliminate a temporary being passed as an owned
// argument to a call.
//
// We currently only do this transformation in extremely limited cases: we
// need to defend against weird situations where "src" doesn't dominate
// "tmp" and where "src" gets re-initialized before the use of "tmp", e.g.:
//
//    %tmp = lit.varlet.decl "anonymous"
//    kgen.call __copyinit__(%tmp, %src)
//    kgen.call __del__(%src)   <<<=== Thinking about inserting this.
//    kgen.call __init__(%src, ...)
//    use(%tmp)
//    use(%src)
//
// Doing this right requires non-trivial liveness analysis which should
// itself be part of a standalone SSA pass post-inlining.  For now we'll
// just catch the most obvious local cases to clean up the IR and provide a
// "guaranteed" optimization.
static bool canEntirelyElideMemoryTemporary(LIT::CallOp copyInitCall,
                                            VarLetDeclOp tmpDecl) {
  Block *tmpBlock = tmpDecl->getBlock();
  if (copyInitCall->getBlock() != tmpBlock)
    return false;

  Value srcPointer = copyInitCall.getOperand(1);

  size_t numUses = 0;
  for (OpOperand &operand : copyInitCall.getOperand(0).getUses()) {
    auto user = dyn_cast<LIT::CallOp>(operand.getOwner());

    // Don't handle control flow or other weird cases that are not calls.
    if (!user || user->getBlock() != tmpBlock)
      return false;

    // Ignore the copyinit.
    if (user == copyInitCall.getOperation())
      continue;
    // We are doing n^2 scanning below, harshly limit it.
    if (++numUses > 3)
      return false;

    // The argument convention for the callee must be consuming, not
    // initializing or anything else.
    auto convention =
        user.getCalleeType().getInputConvention(operand.getOperandNumber());
    if (convention != ValueInputConvention::OwnedInMem)
      return false;

    // Ok, scan to check that nothing between the copyinit and the user of
    // the temp use src.
    for (auto it = ++Block::iterator(copyInitCall), e = tmpBlock->end();;
         ++it) {
      // If we ran off the end of the block, then the copyinit doesn't
      // dominate this use, something weird is going on, bail out.
      if (it == e)
        return false;

      // Scan all the operands to see if any of them are related to %src. We
      // disallow regions because we don't recurse into them.
      if (it->getNumRegions() || llvm::any_of(it->getOperands(), [&](Value v) {
            return mightPointTo(v, srcPointer);
          }))
        return false;

      // If we found the user, then we succeed.  Otherwise keep scanning.
      if (&*it == user.getOperation())
        break;
    }
  }
  return true;
}

/// Given the need to destroy the specified value as a result of the specified
/// operation using it, check to see if the use is a call to the copy ctor for
/// the value.  If so, try to elide the copy+temporary.  This returns success
/// when it can do the elision, failure otherwise.
LogicalResult DestructorInsertion::elideCopyDestroyPair(Value value,
                                                        Type destroyedType,
                                                        Operation *opWithUse) {
  auto copyInitCall = dyn_cast_if_present<LIT::CallOp>(opWithUse);
  if (!copyInitCall)
    return failure();

  // See if we can resolve the callee.
  LIT::FuncOp callee = valueSet.typeDeclInfo.getFuncForSymbol(
      copyInitCall.getCallee().getSymbol());
  if (!callee)
    return failure();

  // Handle the register_passable case:
  //   %newVal = kgen.call __copyinit__(%value) calls.
  //   kgen.call __del__(%value)   <<= Thinking about inserting this.
  //   kgen.call user(%newVal)     <<= Consuming call.
  if (callee.getSpecialFunctionKind() == SpecialFunctionKind::kCopyInitReg) {
    // Make sure the destructor is for the source of the copyinit not the result
    // of the copyinit or something else weird.
    Value srcValue = copyInitCall.getOperand(0);
    if (srcValue != value) {
      // With var's we can have indirect operands.
      bool isOk = false;
      if (auto load = srcValue.getDefiningOp<POP::LoadOp>()) {
        if (load.getOperand() == value) {
          isOk = true;
        } else if (auto refToPointer =
                       load.getOperand().getDefiningOp<RefToPointerOp>()) {
          if (refToPointer.getRef() == value) {
            // TODO(references) remove support for pointers.
            isOk = true;
          }
        }
      } else if (auto load = srcValue.getDefiningOp<LIT::RefLoadOp>()) {
        if (load.getOperand() == value)
          isOk = true;
      }
      if (!isOk)
        return failure();
    }

    // Transform into:
    //   kgen.call user(%value)
    copyInitCall.getResult(0).replaceAllUsesWith(srcValue);
    opsToRemove.push_back(copyInitCall);
    return success();
  }

  // Otherwise handle memory passable copies like:
  //   %tmp = lit.varlet.decl "anonymous"
  //   kgen.call __copyinit__(%tmp, %src)
  //   kgen.call __del__(%src)   <<= Thinking about inserting this.
  //   kgen.call user(%tmp)      <<= Consuming call.
  if (callee.getSpecialFunctionKind() != SpecialFunctionKind::kCopyInit)
    return failure();
  if (copyInitCall.getOperand(1) != value) {
    // TODO(references): remove this.
    auto refToPointer =
        copyInitCall.getOperand(1).getDefiningOp<RefToPointerOp>();
    if (!refToPointer || refToPointer.getRef() != value)
      return failure();
  }

  // We prefer to completely delete the copy if it is into a temporary location
  // that we can forward.
  //
  // Note: we currently delete explicitly declared temporaries, not just
  // implicit ones.  This is a policy decision, and we should look into
  // the impact on debug information, but generally one wouldn't want debug
  // information to block optimizations.
  if (auto refToPtr =
          copyInitCall.getOperand(0).getDefiningOp<RefToPointerOp>()) {
    if (VarLetDeclOp tmpDecl =
            refToPtr.getOperand().getDefiningOp<VarLetDeclOp>()) {
      assert((copyInitCall.getOperand(0).getType() == value.getType() ||
              copyInitCall.getOperand(1).getDefiningOp<RefToPointerOp>()) &&
             copyInitCall.use_empty() && "something strange");

      if (tmpDecl->hasOneUse() &&
          canEntirelyElideMemoryTemporary(copyInitCall, tmpDecl)) {
        refToPtr.getResult().replaceAllUsesWith(copyInitCall.getOperand(1));
        opsToRemove.push_back(copyInitCall);
        opsToRemove.push_back(refToPtr);
        opsToRemove.push_back(tmpDecl);
        return success();
      }
    }
  }

  // Otherwise, try to promote to a __moveinit__/__takeinit__ call if present.
  SymbolConstantAttr moveCtor =
      valueSet.typeDeclInfo.getMoveInitForType(destroyedType);
  if (!moveCtor)
    return failure();

  // moveCtor has two forms: __takeinit__ destructively steals from a live
  // object without destroying it, and __moveinit__ takes and destroys it.  The
  // former takes the operand as inout, the later as owned convention.
  auto moveSig = cast<SignatureType>(moveCtor.getType());
  assert(moveSig.getNumInputs() == 2);
  // TODO(references): reenable this assert when RefToPointerOp is removed.
  // assert(moveSig.getValueInputs()[0] == value.getType() &&
  //       moveSig.getValueInputs()[1] == value.getType());

  // Transform the copy into a move.
  copyInitCall.setCalleeAttr(moveCtor);

  // If this is __moveinit__, then we don't need a dtor call.  If it is
  // __takeinit__, then we need to destroy the husk of the object stolen from.
  if (moveSig.getInputConvention(1) == ValueInputConvention::OwnedInMem)
    return success();
  // We succeeded at the transform, but still need to del.
  return failure();
}

/// Emit one destructor call for one entire value or field.  This should only be
/// called by destroyValueIfNeeded.
///
/// The 'opWithUse' value, if present, is the operation using the overall value
/// being destroyed.  This allows us to perform copy ctor+temp elision.
void DestructorInsertion::emitDestructorCallAt(Value value, ValueRef valueRef,
                                               ImplicitLocOpBuilder &builder,
                                               Operation *opWithUse) {
  // We are going to emit a destructor for the specified ValueRef, so all none
  // of the things we are about to destroy should already be destroyed.
  assert(!dryRun && "this inserts!");
  assert(valueRef.isAllMissing(consumedValues) &&
         "cannot have partially consumed object");

  Type destroyedType = valueRef.getValueType(value);
  TypedAttr dtor = valueSet.typeDeclInfo.getDestructorForType(destroyedType);
  if (!dtor) // Trivial types don't have destructors, so nothing to do.
    return;

  // Okay, if there is a destructor, we know that this is a non-trivial value.
  // Check to see if the operation that we are destroying this for is a
  // copy-ctor.  If so, try to elide the copy constructor: it is better to
  // directly use the original value than to copy it and destroy the original.
  if (succeeded(elideCopyDestroyPair(value, destroyedType, opWithUse)))
    return;

  auto signature = cast<SignatureType>(dtor.getType());
  assert(signature.getNumResults() == 1 &&
         "dtor should have one result (none type)");
  assert(signature.getNumInputs() == 1 && "dtor should have one operand");

  // We may have a @register_passable value indirect (e.g. because it is in a
  // var).  If so, it needs to be loaded to invoke the destructor.
  Value valueToDestroy = value;
  if (auto ref = dyn_cast<RefType>(valueToDestroy.getType())) {
    if (signature.getValueInputs()[0] == ref.getElementAsType())
      valueToDestroy = builder.create<RefLoadOp>(valueToDestroy);
    else
      // TODO(references): pass references to destructors, not pointers.
      valueToDestroy = builder.create<RefToPointerOp>(valueToDestroy);
  }

  // TODO(references): remove this when pointers are gone.
  if (valueToDestroy.getType() != signature.getValueInputs()[0])
    valueToDestroy = builder.create<POP::LoadOp>(valueToDestroy,
                                                 /*align*/ std::nullopt);

  assert(signature.getValueInputs()[0] == valueToDestroy.getType());

  // Emit the call to the destructor.
  if (auto directDtor = dyn_cast<SymbolConstantAttr>(dtor)) {
    builder.create<LIT::CallOp>(signature.getValueResults()[0], directDtor,
                                valueToDestroy);
  } else {
    builder.create<LIT::CallParamOp>(signature.getValueResults()[0], dtor,
                                     std::nullopt, std::nullopt,
                                     valueToDestroy);
  }
}

/// Destroy any values whose bits are indicated in the specified set.  Insert
/// the destructor calls at the entry to the specified block.  This leaves the
/// consumedValues set in an unpredictable state, and is not safe in dryRun
/// mode.
void DestructorInsertion::destroyValuesAtEntry(const BitVector &entries,
                                               Block &block, Location loc) {
  assert(!dryRun && "shouldn't be called in a dry run");

  // Don't bother destroying anything if the block is unreachable.
  if (isa<UnreachableOp>(block.front()))
    return;

  // Any dtor calls will be emitted at the start of the block.
  mlir::ImplicitLocOpBuilder builder(loc, &block, block.begin());

  // We *only* want to destroy the values in entries, not any other values that
  // may be partially overlapped, so mark all the other things as "already
  // destroyed".
  consumedValues = entries;
  consumedValues.flip();

  // As we scan through bits, we walk through corresponding ValueInfos to know
  // what we are working with.
  MutableArrayRef<ValueInfo> valueInfos = valueSet.getValueInfos();
  size_t nextValueInfo = 0;

  int nextToDestroy = entries.find_first();
  while (nextToDestroy != -1) {
    // Figure out which valueInfo this is.
    while (!valueInfos[nextValueInfo].contains(nextToDestroy)) {
      ++nextValueInfo;
      assert(nextValueInfo != valueInfos.size() &&
             "nothing contains this bit?");
    }

    // Ok, we know that we are destroying some field of this value, find the
    // whole value so we know the MLIR value.
    ValueRef fullValueRef = valueSet.getFullValueRef(nextValueInfo);

    // Emit destructor calls for the entire value or the correct subfields that
    // need to be destroyed.
    destroyValueIfNeeded(valueInfos[nextValueInfo].value, fullValueRef, builder,
                         /*opWithUse=*/nullptr);

    // Find the next object to destroy.
    nextToDestroy = entries.find_next(fullValueRef.endBit - 1);
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
    // Find all the functions and structs in the module.
    auto [functionVector, funcMap, structMap, traitMap] =
        collectFunctionsAndTypes(getOperation());

    // Process all the structs into TypeDeclInfo.
    TypeDeclInfo typeDeclInfo(std::move(structMap), std::move(funcMap),
                              std::move(traitMap));

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
  ValueSet valueSet(typeDeclInfo, func);

  /// This is set to true if the function has nested functions inside of it.
  /// Some of our analyses are not safe in the face of closures yet.
  bool hasClosures = false;

  func->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // Skip looking at nested functions, they are handled as separate contexts.
    if (op != func && isa<LIT::FuncOp>(op)) {
      hasClosures = true;
      return WalkResult::skip();
    }

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
  SmallVector<Operation *> opsToRemove;
  DestructorInsertion(valueSet, opsToRemove).scanFunction(func);

  // Now that we've looked at all the uses and definitions in the function,
  // diagnose any 'var's that should be written as 'let's with a warning.
  //
  // FIXME: Our analysis is not safe in the presence of closures, so disable
  // this check for now if we see any.
  if (!hasClosures)
    for (const ValueInfo &info : valueSet.getValueInfos()) {
      if (info.hasErrorDiagnosed || info.isMutatedWhenInitialized ||
          !info.value)
        continue;

      auto checkVarLet = [&](auto varLet) {
        if (varLet.getKind() != VarLetDeclKind::Var)
          return;
        mlir::emitWarning(varLet.getLoc())
            << "'" << varLet.getName()
            << "' was declared as a 'var' but never mutated, consider "
               "switching to a 'let'";
      };

      if (auto varLet = info.value.getDefiningOp<VarLetDeclOp>())
        checkVarLet(varLet);
    }

  // Remove copy ctors and allocations that have been elided.
  for (Operation *op : opsToRemove)
    op->erase();

  // Return failure if we generated errors for any of the tracked values.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  }));
}
