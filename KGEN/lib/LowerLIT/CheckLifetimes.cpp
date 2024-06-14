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
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

// Targets that are outside the ControlFlowNode. Negative numbers indicate
// outside of op. Positive numbers are indices into regions of control flow
// node.
enum ControlFlowTarget {
  ParentPrev = -1,
  ParentPost = -2,
  Continue = -3,
  Break = -4,
  Raise = -5,
  Return = -6
};

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
      if (!funcOp.isOptionalSymbol())
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

/// Create DebugInfo::DILocalVariableAttr if this VarDecl needs it.
/// `funcSpAttr` is the DISubprogramAttr of the surrounding function.
static DebugInfo::DILocalVariableAttr
createDebugVariableForVarDecl(VarDeclOp op,
                              DebugInfo::DISubprogramAttr funcSpAttr) {
  if (op.getKind() == VarDeclKind::Synthesized)
    return {};

  Location loc = op->getLoc();
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return {};

  auto localScope = DebugInfo::extractScopeFrom<DebugInfo::DILocalScopeAttr>(
      loc, DebugInfo::LocWalkPolicy::CalleePriority);
  if (!localScope)
    return {};

  // The source type is the decl type with ref unwrapped.
  auto sourceType =
      DebugInfo::DIUnresolvedMLIRType::get(op.getType().getElementType());
  auto varAttr = DebugInfo::DILocalVariableAttr::get(
      localScope, op.getNameAttr(), funcSpAttr.getFile(), fileLoc.getLine(),
      /*arg=*/op.getArgShadowIndex().value_or(-1) + 1,
      /*alignInBits=*/0, sourceType);

  return varAttr;
}

/// Inserts a DebugInfo::ValueOp for this block argument if necessary.
/// `funcSpAttr` is the DISubprogramAttr of the surrounding function `func`.
/// Returns the VarInfo of the inserted ValueOp.
static DebugInfo::DILocalVariableAttr
insertDebugVariableForArg(OpBuilder &builder, LIT::FuncOp func,
                          BlockArgument arg, ArrayRef<PogMetadataAttr> pogList,
                          DebugInfo::DISubprogramAttr funcSpAttr) {
  // Skip synthesized args.
  if (arg.getArgNumber() >= pogList.size())
    return {};

  StringRef name = pogList[arg.getArgNumber()].getName();
  if (name.empty())
    return {};

  Location loc = arg.getLoc();
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return {};

  DebugInfo::DIType sourceType;
  DebugInfo::DIExprAttr diExpr;
  ArgConvention convention =
      func.getSignature().getArgConvention(arg.getArgNumber());
  if (SignatureType::hasAddress(convention)) {
    // If this argument has address, its source type is the raw type.
    if (auto argRefType = dyn_cast<RefType>(arg.getType())) {
      sourceType =
          DebugInfo::DIUnresolvedMLIRType::get(argRefType.getElementType());
      auto diPointerType =
          DebugInfo::DITargetIndependentPointerType::get(sourceType);
      auto newIrValue = DebugInfo::DIIRValueExprAttr::get(diPointerType);
      diExpr = DebugInfo::DIDerefExprAttr::get(newIrValue);
    }
  }

  if (!sourceType) {
    // Otherwise, its source type is the arg type itself.
    sourceType = DebugInfo::DIUnresolvedMLIRType::get(arg.getType());
    diExpr = DebugInfo::DIIRValueExprAttr::get(sourceType);
  }

  DebugInfo::DILocalVariableAttr varAttr = DebugInfo::DILocalVariableAttr::get(
      funcSpAttr, name, funcSpAttr.getFile(), fileLoc.getLine(),
      arg.getArgNumber() + 1,
      /*alignInBits=*/0, sourceType);
  auto scopedLoc =
      FusedLoc::get(varAttr.getContext(), {loc}, varAttr.getScope());

  builder.create<DebugInfo::ValueOp>(scopedLoc, arg, varAttr, diExpr);
  return varAttr;
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
  if (DeclRefType valueType = dyn_cast<DeclRefType>(type))
    return getStructDeclForType(valueType).isRegisterPassableTrivial();

  // Other values of raw MLIR type are always trivial.
  return true;
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
  auto newSig = attr.getType().getSpecializedSignature(paramValues);
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
        // Bind the *(0,0) parameter to a concrete type we're using in this
        // context.
        auto specSig = dtorSig.getSpecializedSignature({generic.getParam()});
        auto delStr =
            StringAttr::get("__del__", StringType::get(type.getContext()));
        return ParamOperatorAttr::get(POC::GetTypeMethod,
                                      {generic.getParam(), delStr}, specSig);
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
  /// Enum indicating whether the value is initalized at function exit.
  const LifetimeTrackable::ExitInitState endInitState;

  /// True if this value lives in memory, not a @register_passable SSA value.
  const bool isIndirect;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  const bool isFullObjectLiveOnEntry;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;

  /// If this value needs to be tracked by debug info, this is the information
  /// about the source variable that created this value. Null otherwise.
  DebugInfo::DILocalVariableAttr debugVariable;

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

  static Type getDereferencedType(Type sourceTy, bool isIndirect) {
    // If this is a direct value, use the type directly.
    return isIndirect ? cast<RefType>(sourceTy).getElementType() : sourceTy;
  }

  /// Return the type of the underlying value, looking through the reference
  /// type if indirect.
  Type getValueType(Value value) const {
    return getDereferencedType(value.getType(), isIndirect);
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
  ///
  /// This sentinel is also used by DestructorInsertion as a marker for
  /// "unreachable" code to avoid unnecessary meets.
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
  ///  * whether the value is a by-ref to the underlying logical value
  ///  * whether the value starts out uninit or init at the function start
  ///  * whether the value is uninit or init at normal function return.
  void addValue(Value val, const LifetimeTrackable &trackable,
                DebugInfo::DILocalVariableAttr debugVariable = {}) {
    // Figure out how many bits to track for this value at the lifetime if mem.
    unsigned numValueBits;
    TypedAttr valueLifetime;
    if (!val) {
      numValueBits = 1; // Nothing to do for the sentinel.
    } else if (trackable.isIndirect) {
      // This should be an assertion, but check softly to help IR clients.
      auto refType = dyn_cast<RefType>(val.getType());
      if (!refType) {
        mlir::emitError(val.getLoc())
            << "trackable IR value of type " << val.getType()
            << " should have type '!lit.ref': " << val;
        return;
      }
      Type valType = refType.getElementType();
      numValueBits = typeDeclInfo.getNumFieldsInType(valType);

      // Remember the lifetime if not immortal.
      if (!isa<LifetimeAttr>(refType.getLifetime()))
        valueLifetime = refType.getLifetime();
    } else {
      // We don't track trivial values of register type.
      if (typeDeclInfo.isRegisterPassableTrivial(val.getType()))
        return;
      // We are only field sensitive for memory objects, not in-register values.
      numValueBits = 1;
    }
    unsigned firstValueBit = getNumTotalBits();

    // Record this information in our tables.
    valueInfoIndex[val] = valueInfos.size();
    if (valueLifetime)
      lifetimeToValueIndex[valueLifetime] = valueInfos.size();

    valueInfos.push_back({val, firstValueBit, firstValueBit + numValueBits,
                          trackable.startsUninit, trackable.endInitState,
                          trackable.isIndirect,
                          trackable.isFullObjectLiveOnEntry,
                          /*hasErrorDiagnosed=*/false, debugVariable});
  }

  /// Return a reference to the entire value with the specified ID.
  ValueRef getFullValueRef(unsigned valueId) const {
    const auto &entry = valueInfos[valueId];
    return ValueRef{valueId, entry.startValueBit, entry.endValueBit,
                    entry.isIndirect};
  }

  /// Given a lifetime attribute, return the value ref that defines it.
  ValueRef getFullValueRefForLifetime(TypedAttr lifetime) const {
    auto it = lifetimeToValueIndex.find(lifetime);
    if (it == lifetimeToValueIndex.end())
      return {};
    return getFullValueRef(it->second);
  }

  /// Look up all the value refs that an access with the specified Value and
  /// dereference bit touch.
  SmallVector<ValueRef> getValueRefsForAccess(Value val, bool isDeref);
  SmallVector<ValueRef> getValueRefsForLifetime(TypedAttr lifetime);

  /// Given a tracked value that is being accessed by an operation, return
  /// the ValueRef for the object being tracked or null if untracked.
  ///
  /// 'isDeref' indicates that this is an indirect use of the specified value,
  /// which matters in the case of references.  When false, this is a use of a
  /// possibly-owned register value.
  ValueRef getDirectValueRef(Value value, bool isDeref) const;

  /// Return the total number of bits we need to track in the bitvector.
  unsigned getNumTotalBits() const {
    return !valueInfos.empty() ? valueInfos.back().endValueBit : 0;
  }

  /// Return true if this reference is to a trivial value that is not tracked
  /// for liveness.
  bool isTrivial(Type type, bool isIndirect) const;
  bool isTrivial(Value value, bool isIndirect) const {
    return isTrivial(value.getType(), isIndirect);
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
  /// This is the function we're analyzing.
  LIT::FuncOp func;
  /// These are all of the value infos, indexed by ID #.
  SmallVector<ValueInfo> valueInfos;
  /// This is a lookup from SSA values to the thing they are referencing.
  DenseMap<Value, unsigned> valueInfoIndex;
  /// This is a mapping of lifetime attrs to the value index that defines them.
  DenseMap<TypedAttr, unsigned> lifetimeToValueIndex;
};
} // namespace

bool ValueSet::isTrivial(Type type, bool isIndirect) const {
  auto eltType = ValueRef::getDereferencedType(type, isIndirect);
  return typeDeclInfo.isRegisterPassableTrivial(eltType);
}

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
    switch (info.endInitState) {
    case LifetimeTrackable::EndsInit:
      break;
    case LifetimeTrackable::EndsUninit:
      os << " EI";
      break;
    case LifetimeTrackable::InitOnNormal:
      os << " NR";
      break;
    case LifetimeTrackable::InitOnError:
      os << " ER";
      break;
    }
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
///
/// 'isDeref' indicates that this is an indirect use of the specified value,
/// which matters in the case of references.  When false, this is a use of a
/// possibly-owned register value.
ValueRef ValueSet::getDirectValueRef(Value value, bool isDeref) const {
  // If the value is deref, it must have reference type.
  assert((!isDeref || isa<RefType>(value.getType())) &&
         "only references are dereferencable!");

  // If this is testing a reference value (not the dereference value) then it is
  // ignored: references can be passed around and used with the contents being
  // liveness tracked, the ultimate accesses are what matter.
  if (!isDeref && isa<RefType>(value.getType()))
    return {};

  // If this is a value we're tracking, return it.
  auto it = valueInfoIndex.find(value);
  if (it != valueInfoIndex.end())
    return getFullValueRef(it->second);

  // If this is a GER, check the base and focus in on a field of it.
  if (auto structGER = value.getDefiningOp<RefStructGEROp>()) {
    ValueRef baseVal = getDirectValueRef(structGER.getContainer(), isDeref);
    if (!baseVal || !baseVal.isIndirect)
      return {};

    // Figure out what subset of elements we have indexed to.
    auto containerType = structGER.getContainer().getType().getElementType();
    unsigned fieldOffset = typeDeclInfo.getFieldIndex(
        cast<DeclRefType>(containerType), structGER.getFieldAttr());
    unsigned startBit = baseVal.startBit + fieldOffset;
    auto resultType = structGER.getType().getElementType();
    return ValueRef{baseVal.valueId, startBit,
                    startBit + typeDeclInfo.getNumFieldsInType(resultType),
                    /*isIndirect=*/true};
  }

  if (auto load = value.getDefiningOp<RefLoadOp>())
    if (auto valueRef = getDirectValueRef(load.getRef(), /*isDeref=*/true)) {
      if (valueRef.isIndirect) {
        // The parser doesn't emit all the lifetime stuff for trivial types,
        // so don't track them either.
        if (isTrivial(load, /*isDeref=*/false))
          return {};

        valueRef.isIndirect = false;
        return valueRef;
      }
    }

  // If this is a RebindOp get the underlying ref.
  if (auto rebind = value.getDefiningOp<RebindOp>())
    return getDirectValueRef(rebind.getOperand(), /*isDeref=*/isDeref);
  if (auto immut = value.getDefiningOp<RefImmutOp>())
    return getDirectValueRef(immut.getOperand(), /*isDeref=*/isDeref);

  // Otherwise, we don't know what this is.
  return ValueRef();
}

/// Look up all the value refs that an access to the specified lifetime could
/// touch.
SmallVector<ValueRef> ValueSet::getValueRefsForLifetime(TypedAttr lifetime) {
  SmallVector<ValueRef> result;

  // FIXME: Track mutability correctly.
  lifetime = LifetimeMutCastAttr::strip(lifetime);

  // If the lifetime is a LifetimeUnionAttr then it will already be uniqued,
  // inlined, and stripped of immortal references, so we can just return all
  // the value refs for its elments.
  if (auto unionAttr = dyn_cast<LifetimeUnionAttr>(lifetime)) {
    for (auto elt : unionAttr.getOperands())
      if (auto valueRef =
              getFullValueRefForLifetime(LifetimeMutCastAttr::strip(elt)))
        result.push_back(valueRef);
  } else if (auto valueRef = getFullValueRefForLifetime(lifetime))
    result.push_back(valueRef);
  return result;
}

/// Look up all the value refs that an access with the specified Value and
/// dereference bit touch.
SmallVector<ValueRef> ValueSet::getValueRefsForAccess(Value value,
                                                      bool isDeref) {
  // If this is a direct reference to a value, return field sensitive info.
  if (ValueRef valueRef = getDirectValueRef(value, isDeref)) {
    SmallVector<ValueRef> result;
    result.push_back(valueRef);
    return result;
  }

  // Otherwise, if indirect, this is an reference to one or more
  // lifetime-tracked values, figure out what they are.
  if (isDeref)
    return getValueRefsForLifetime(
        cast<RefType>(value.getType()).getLifetime());

  // Otherwise it is a trivial or untracked value.
  return {};
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
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkAcyclicControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  void diagnoseUsageError(ValueRef valueRef, Operation &op, bool isDef);
  void checkUse(Value value, Operation &op, bool isDeref);
  void checkDef(Value value, Operation &op, bool isDeref);
  void checkConsume(Value value, Operation &op, bool isDeref);
  void checkMarkDestroyed(Value value, Operation &op);
  void checkLifetimeEffect(TypedAttr lifetime, Operation &op);

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
  /// When analyzing the body of a try, this bitset indicates what a
  /// 'raise' should intersect with.
  BitVector *raiseSet = nullptr;
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

/// Verify that the specified ValueRef is live at this point, diagnosing an
/// error at the specified operation if not.
void UninitializedValueScan::checkUse(Value value, Operation &op,
                                      bool isDeref) {
  SmallVector<ValueRef> accesses =
      valueSet.getValueRefsForAccess(value, isDeref);

  for (ValueRef access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveValues))
      diagnoseUsageError(access, op, /*isDef=*/false);
  }
}

/// One of the specified fields is missing, so emit an error about it.  This is
/// largely to complain about incorrect 'uses' of a value, but When
/// 'isDef' is true this is complaining about an indirect def of a value.
void UninitializedValueScan::diagnoseUsageError(ValueRef valueRef,
                                                Operation &op, bool isDef) {
  // Ok, it isn't, gear up to see how to best report the error.
  ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);
  if (valueEntry.hasErrorDiagnosed)
    return; // Only report one error per symbolic value.
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
    return;
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
    if (!isDef)
      diag << "use of uninitialized value ";
    else
      diag << "potential indirect access to uninitialized value ";

    // If some fields are present and others are missing, complain about the
    // first whole field that is missing.
    addBadValueNameToDiag(valueRef, liveValues, valueSet, diag);
  }
  diag.attachNote(valueEntry.value.getLoc())
      << "'" << valueEntry.getName().str() << "' declared here";
}

void UninitializedValueScan::checkDef(Value value, Operation &op,
                                      bool isDeref) {
  // Direct accesses are handled in a field sensitive way, and this can count as
  // an initialization.
  if (ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref)) {
    // Finally, marks its value live so any use after this isn't treated as
    // uninitialized.
    valueRef.markBits(liveValues, true);
    return;
  }

  // If this is an indirect reference then a mutation will require that all
  // values being mutated are initialized, because we cannot perform field
  // sensitive initialization, only overwrite/mutate.
  SmallVector<ValueRef> accesses =
      valueSet.getValueRefsForAccess(value, isDeref);
  for (auto access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveValues))
      diagnoseUsageError(access, op, /*isDef=*/true);
  }
}

void UninitializedValueScan::checkConsume(Value value, Operation &op,
                                          bool isDeref) {
  ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref);
  if (!valueRef) {
    // We cannot consume an indirect value (unless it is untracked).
    if (!valueSet.isTrivial(value, isDeref) &&
        // FIXME(#29005): AnyRefType binds to non-trivial types
        isDeref) {
      mlir::emitError(op.getLoc(),
                      "cannot consume indirect references to values");
    }
    return;
  }

  // The value must be completely live in order for us to consume it.  If not,
  // use "checkUse" to diagnose the problem.
  if (!valueRef.isAllPresent(liveValues))
    diagnoseUsageError(valueRef, op, /*isDef*/ false);

  // If tracked, marks its value as dead.
  if (!valueSet.isTrivial(value, isDeref))
    valueRef.markBits(liveValues, false);
}

/// The lit.ownership.mark_destroyed op consumes the whole object bit of
/// a value only, but not its fields.
void UninitializedValueScan::checkMarkDestroyed(Value value, Operation &op) {
  SmallVector<ValueRef> accesses =
      valueSet.getValueRefsForAccess(value, /*isDeref=*/true);

  auto numBitsForAccess = valueSet.typeDeclInfo.getNumFieldsInType(
      cast<RefType>(value.getType()).getElementType());

  for (auto valueRef : accesses) {
    // Make sure only whole-values are being referenced, not subfields.
    ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
    if (info.endValueBit - info.startValueBit != numBitsForAccess) {
      if (!info.hasErrorDiagnosed) {
        mlir::emitError(op.getLoc(), "cannot mark subobjects destroyed");
        info.hasErrorDiagnosed = true;
      }
      return;
    }

    // Check that the consumed bit is live, otherwise it cannot be destroyed.
    valueRef = valueRef.getSubfield(valueRef.getNumBits() - 1, 1);

    // If not, then there is an error which we diagnose.
    if (!valueRef.isAllPresent(liveValues))
      diagnoseUsageError(valueRef, op, /*isDef=*/false);
  }
}

/// Check any unstructured lifetimes that are accessed by the operation.
void UninitializedValueScan::checkLifetimeEffect(TypedAttr lifetime,
                                                 Operation &op) {
  // We assume this may mutate the lifetime unless we know it is read-only.
  bool isMutate = !cast<LifetimeType>(lifetime.getType()).isMutableKnown(false);

  SmallVector<ValueRef> accesses = valueSet.getValueRefsForLifetime(lifetime);
  for (auto access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveValues))
      diagnoseUsageError(access, op, /*isDef=*/isMutate);
  }
}

void UninitializedValueScan::scanFunction(LIT::FuncOp func) {
  // Initialize the BitVector with all the elements that are live-in.  We treat
  // all values live at the start of the function (even before they are actually
  // defined) because we know that all uses must be after them due to SSA
  // dominance.
  liveValues.resize(valueSet.getNumTotalBits());
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
  SmallVector<std::pair<Value, OperandEffect>> operandEffects;
  SmallVector<ResultEffect> resultEffects;
  SmallVector<TypedAttr> lifetimeEffects;
  for (Operation &op : block) {
    operandEffects.clear();
    resultEffects.clear();
    lifetimeEffects.clear();
    auto overall =
        getOperationEffects(op, operandEffects, resultEffects, lifetimeEffects);
    /// If the operation is unknown, ignore it.
    if (overall == OverallOpValueEffect::unknownOp) {
      // NOTE: Can log here when extending things.
      // op.dump();
      continue;
    }

    assert(resultEffects.size() == op.getNumResults() &&
           "getOperationEffects returned wrong # effects");

    // Handle all the normal operand and result effects.
    for (auto [operand, effect] : operandEffects) {
      switch (effect) {
      case OperandEffect::regUse:
        checkUse(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::regConsume:
        checkConsume(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::memLoad:
        checkUse(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memStoreOwned:
        checkDef(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memInOut:
        checkUse(operand, op, /*isDeref=*/true);
        checkDef(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memConsume:
        checkConsume(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memMarkDestroyed:
        checkMarkDestroyed(operand, op);
        break;
      case OperandEffect::memStoreConditional:
        // This should have been handled by `checkIfLikeOp` above.
        break;
      }
    }

    for (auto [result, effect] : llvm::zip(op.getResults(), resultEffects)) {
#ifndef NDEBUG
      LifetimeTrackable trackable(result);
      // Perform some general sanity checking of the LifetimeTrackable
      // implementation.

      // Since this is an op result, the live in/out behavior must match each
      // other: if this weren't true, then control flow paths that didn't cross
      // the op could never be satisfied.
      bool endsUninit = false;
      if (trackable) {
        assert((trackable.endInitState == LifetimeTrackable::EndsInit ||
                trackable.endInitState == LifetimeTrackable::EndsUninit) &&
               "invalid end init state for an op result");
        endsUninit = trackable.endInitState == LifetimeTrackable::EndsUninit;
        assert(trackable.startsUninit == endsUninit &&
               "op results must have same live in/out behavior");
      }
#endif

      switch (effect) {
      case ResultEffect::ignore:
        assert(!trackable && "Lifetime trackable and CheckLifetimes disagree");
        continue;
      case ResultEffect::regDefine:
        assert(trackable && !trackable.isIndirect && endsUninit &&
               "Lifetime trackable and CheckLifetimes disagree");
        checkDef(result, op, /*isDeref=*/false);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by LifetimeTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Lifetime trackable and CheckLifetimes disagree");
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true);
        break;
      case ResultEffect::memDefineUninitToUninit:
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Lifetime trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Lifetime trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToUninit:
        // The live-in behavior is modeled by LifetimeTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Lifetime trackable and CheckLifetimes disagree");
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true);
        break;
      }
    }

    // Process any other indirect lifetimes accessed.
    for (auto lifetime : lifetimeEffects)
      checkLifetimeEffect(lifetime, op);

    switch (overall) {
    case OverallOpValueEffect::unknownOp:
    case OverallOpValueEffect::allHandled:
      // No special action.
      break;
    case OverallOpValueEffect::terminatorOp:
      checkTerminatorOp(op);
      break;
    case OverallOpValueEffect::localControlFlowOp:
      checkLocalControlFlowOp(op);
      break;
    case OverallOpValueEffect::ifLikeOp:
      checkIfLikeOp(op);
      break;
    case OverallOpValueEffect::acyclicControlFlowNodeOp:
      checkAcyclicControlFlowOp(op);
      break;
    case OverallOpValueEffect::loopOp:
      checkLoopOp(op);
      break;
    case OverallOpValueEffect::tryOp:
      checkTryOp(cast<LIT::TryOp>(op));
      break;
    }
  }
}

/// Return true if the value is uninitialized at the given exit from the
/// function. A value may be always uninitialized or initialized, or it may be
/// depending on the exit kind.
static bool isUninitializedAtExit(const ValueInfo &valueInfo, Operation &exit) {
  return (valueInfo.endInitState == LifetimeTrackable::EndsUninit) ||
         (valueInfo.endInitState == LifetimeTrackable::InitOnNormal &&
          isa<ErrorReturnOp>(exit)) ||
         (valueInfo.endInitState == LifetimeTrackable::InitOnError &&
          isa<KGEN::ReturnOp>(exit));
}

/// This is called when the op is a return, lit.error_return or unreachable op.
void UninitializedValueScan::checkTerminatorOp(Operation &op) {
  // If this is a kgen.return then we have an exit from the function
  // (including early returns and exception raises that leave the function).
  // Check that *all* of the values are live-out of the function are
  // initialized.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op)) {
    for (const ValueInfo &valueInfo :
         llvm::drop_begin(valueSet.getValueInfos())) {
      // If the value doesn't need to be live at end of function, ignore it.
      if (isUninitializedAtExit(valueInfo, op))
        continue;

      // If this is the hacky RefFromPointerREPLOp op (used by the REPL
      // only!) and if this is an error path, then we look the other way at
      // indiscretions.
      if (valueInfo.value.getDefiningOp<RefFromPointerREPLOp>() &&
          isa<LIT::ErrorReturnOp>(op))
        continue;

      // Otherwise, it must be live at return/raise.
      checkUse(valueInfo.value, op, /*isDeref=*/valueInfo.isIndirect);
    }
  } else {
    assert(isa<KGEN::UnreachableOp>(op) && "Unknown terminator");
  }

  // Indicate that all values are live after the return so that an early
  // return in an 'if' will get properly intersected with the other side
  // of the branch.
  liveValues.set();
}

/// This is HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, which all
/// perform local control flow.
void UninitializedValueScan::checkLocalControlFlowOp(Operation &op) {
  if (isa<HLCF::BreakOp, ParamForBreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    *breakSet &= liveValues;
  } else if (isa<HLCF::ContinueOp, ParamForContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    *continueSet &= liveValues;
  } else {
    assert(isa<LIT::TryRaiseOp>(op) && "Unknown local CF op");
    assert(raiseSet && "Not in a 'try'?");
    *raiseSet &= liveValues;
  }

  // Indicate that all values are live after the terminator so an 'if' will get
  // properly intersected with the other side of the branch.
  liveValues.set();
}

void UninitializedValueScan::checkAcyclicControlFlowOp(Operation &op) {
  auto controlFlowNode = cast<HLCF::ControlFlowNode>(op);
  std::optional<BitVector> resultValues;

  SmallVector<std::pair<HLCF::ControlFlowTarget, BitVector>> liveValuesAtTarget;
  SmallVector<HLCF::ControlFlowTarget> targets;
  SmallVector<Attribute> controlFlowNodeOperands;
  for (unsigned i = 0, e = controlFlowNode->getNumOperands(); i < e; i++)
    controlFlowNodeOperands.push_back(Attribute());
  controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
  assert(!targets.empty() && "expected at least 1 target to enter op");
  for (HLCF::ControlFlowTarget target : targets)
    liveValuesAtTarget.emplace_back(
        std::pair<HLCF::ControlFlowTarget, BitVector>(target, liveValues));
  while (!liveValuesAtTarget.empty()) {
    auto [target, localLiveValues] = liveValuesAtTarget.back();
    liveValuesAtTarget.pop_back();
    if (target.index.has_value()) {
      unsigned index = target.index.value();
      BitVector initialLiveValues = localLiveValues;
      Region &targetRegion = op.getRegion(index);

      // Compute Liveness after this target.
      liveValues = initialLiveValues;
      scanBlock(targetRegion.front());
      BitVector postLiveValues = liveValues;

      // Advance to next target.
      HLCF::ControlFlowTerminator term = cast<HLCF::ControlFlowTerminator>(
          targetRegion.front().getTerminator());
      if (!term.isParentNode(controlFlowNode) || isa<UnreachableOp>(term)) {
        // Path has completed.
        if (resultValues.has_value())
          resultValues.value() &= postLiveValues;
        else
          resultValues = postLiveValues;
        continue;
      }
      SmallVector<HLCF::ControlFlowTarget> termTargets;
      SmallVector<Attribute> operands(term->getNumOperands());
      term.getBranchTargets(operands, termTargets);
      for (HLCF::ControlFlowTarget t : termTargets) {
        liveValuesAtTarget.emplace_back(
            std::pair<HLCF::ControlFlowTarget, BitVector>(t, postLiveValues));
      }
    } else {
      // Path has completed.
      if (resultValues.has_value())
        resultValues.value() &= localLiveValues;
      else
        resultValues = localLiveValues;
    }
  }
  assert(resultValues.has_value() && "expected at least 1 completed path");
  liveValues = resultValues.value();
}

/// This is HLCF::IfOp, ParamIfOp, or a throwing call, which are all if-like.
void UninitializedValueScan::checkIfLikeOp(Operation &op) {
  // 'if' operations treat the condition as a use but have live outs that are
  // the intersection of the live values produced by the then/else branches.
  assert((isa<HLCF::IfOp, ParamIfOp>(op)));
  assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
         op.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");

  BitVector liveValuesCopy = liveValues;
  scanBlock(op.getRegion(0).front());
  liveValuesCopy.swap(liveValues);
  scanBlock(op.getRegion(1).front());
  liveValues &= liveValuesCopy;
}

void UninitializedValueScan::checkLoopOp(Operation &loopOp) {
  UninitializedValueScan bodySets(valueSet);
  // Loops are transparent to raise.
  bodySets.raiseSet = raiseSet;

  // The default continueSet is the live-in set of values.  This can lose
  // values if some 'continue' path through the body of the loop consumes a
  // value.
  BitVector continueSet(liveValues);
  bodySets.continueSet = &continueSet;

  // The 'breakSet' of the loop body will be the live outs of the loop.  We
  // need to start it out thinking that everything is live so intersections
  // from the body work correctly.
  BitVector breakSet(liveValues.size(), true);
  bodySets.breakSet = &breakSet;

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
    bodySets.scanBlock(loopOp.getRegion(0).front());

    // If any bits got cleared from the continueSet then we need to iterate.
  } while (continueSet.count() != numLiveIn);
  // Any code after the loop continues on with the breaks valid.

  // If the loop has an 'else' region, scan it and then intersect with the loop
  // region.
  if (loopOp.getNumRegions() == 2) {
    scanBlock(loopOp.getRegion(1).front());
    liveValues &= breakSet;
  } else {
    liveValues = std::move(breakSet);
  }
}

void UninitializedValueScan::checkTryOp(LIT::TryOp tryOp) {
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
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op);
  void checkAcyclicControlFlowOp(Operation &op);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  struct BlockConsumeInfo {
    const BitVector &consumedValues;
    Block &block;
  };
  void unifyConsumedSets(Operation &condOp, Block &consumedValueBlock,
                         BlockConsumeInfo otherBlockInfo);
  void checkConsume(Value value, Operation &op, bool isDeref);
  void checkUse(Value value, Operation &op, bool isDeref);
  void checkUse(Value value, mlir::ImplicitLocOpBuilder &builder,
                Operation *opWithUse, bool isDeref);
  void checkDef(Value value, Operation &op, bool isDeref,
                bool needsCheckUse = true);
  void checkLifetimeEffect(TypedAttr lifetime, Operation &op);
  void destroyValuesAtEntry(const BitVector &entries, Block &block,
                            Location loc);
  void destroyValueIfNeeded(Value value, ValueRef valueRef,
                            mlir::ImplicitLocOpBuilder &builder,
                            Operation *opWithUse);

  LogicalResult elideCopyDestroyPair(Value value, Type destroyedType,
                                     Operation *opWithUse);
  void emitDestructorCallAt(Value value, bool isIndirect,
                            mlir::ImplicitLocOpBuilder &builder,
                            Operation *opWithUse);

  /// Emit a debug value for the value if it is tracked with debug info.
  void emitDebugInit(Value value, ValueRef valueRef,
                     mlir::ImplicitLocOpBuilder &builder);

  /// Emit a debug kill marker for the value if it is tracked with debug info.
  void emitDebugKill(ValueRef valueRef, mlir::ImplicitLocOpBuilder &builder);

  /// Emit both a debug kill & a destructor call.
  void emitDebugKillAndDestructorCallAt(Value value, ValueRef valueRef,
                                        mlir::ImplicitLocOpBuilder &builder,
                                        Operation *opWithUse);

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

void DestructorInsertion::dump() const {
  auto &os = llvm::errs();
  if (valueSet.getValueInfos().size() < 32) {
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
  // Slot 0 indicates this block is reachable.  This will be cleared if an
  // 'unreachable' operation is noticed.
  consumedValues.set(0);

  // Scan the body of the function.
  Block &funcBody = func.getFunctionBody().front();
  scanBlock(funcBody);

  // The sentinel tracks reachability.
  assert(consumedValues[0] && "function entry should be reachable");

  // If any argument values are unconsumed then they must be unused.
  // Emit their destructor calls at the start of the function by acting as
  // though there is a use.
  for (auto [argValue, conv] : llvm::zip(
           func.getArguments(), func.getSignature().getArgConventions())) {
    // Ignore undef-on-input values.
    if (SignatureType::isResultSlot(conv) || conv == ArgConvention::InitSelf)
      continue;

    bool isIndirect = SignatureType::hasAddress(conv);
    Location loc = argValue.getLoc();
    if (DebugInfo::DISubprogramAttr scope =
            DebugInfo::extractScope(cast<mlir::FunctionOpInterface>(*func)))
      loc = FusedLoc::get(loc.getContext(), {loc}, scope);

    mlir::ImplicitLocOpBuilder builder(loc, &funcBody, funcBody.begin());
    checkUse(argValue, builder, /*opWithUse=*/nullptr, /*isDeref=*/isIndirect);
  }
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void DestructorInsertion::scanBlock(Block &block) {
  // Process each operation bottom-up in the block.
  SmallVector<std::pair<Value, OperandEffect>> operandEffects;
  SmallVector<ResultEffect> resultEffects;
  SmallVector<TypedAttr> lifetimeEffects;
  for (Operation &op : llvm::reverse(block)) {
    operandEffects.clear();
    resultEffects.clear();
    lifetimeEffects.clear();
    auto overall =
        getOperationEffects(op, operandEffects, resultEffects, lifetimeEffects);
    switch (overall) {
    case OverallOpValueEffect::unknownOp:
      // NOTE: Enable logging when debugging.
      // op.dump();
      continue;
    case OverallOpValueEffect::allHandled:
      break; // No special action.
    case OverallOpValueEffect::terminatorOp:
      checkTerminatorOp(op);
      break;
    case OverallOpValueEffect::localControlFlowOp:
      checkLocalControlFlowOp(op);
      break;
    case OverallOpValueEffect::ifLikeOp:
      checkIfLikeOp(op);
      break;
    case OverallOpValueEffect::acyclicControlFlowNodeOp:
      checkAcyclicControlFlowOp(op);
      break;
    case OverallOpValueEffect::loopOp:
      checkLoopOp(op);
      break;
    case OverallOpValueEffect::tryOp:
      checkTryOp(cast<LIT::TryOp>(op));
      break;
    }

    assert(resultEffects.size() == op.getNumResults() &&
           "getOperationEffects returned wrong # effects");

    for (auto [result, effect] : llvm::zip(op.getResults(), resultEffects)) {
      // CheckUninit pass does all the paranoid checking, don't duplicate it.
      switch (effect) {
      case ResultEffect::ignore:
        continue;
      case ResultEffect::regDefine:
        checkDef(result, op, /*isDeref=*/false);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by LifetimeTrackable to match the
        // live-out behavior.
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true);
        break;
      case ResultEffect::memDefineUninitToUninit:
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        // If it start/end initialized, emit destructor if already replaced.
        checkUse(result, op, /*isDeref=*/true);
        break;
      case ResultEffect::memDefineInitToUninit:
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true);
        break;
      }
    }

    // Handle all the normal operand and result effects.
    for (auto [operand, effect] : operandEffects) {
      switch (effect) {
      case OperandEffect::regUse:
        checkUse(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::regConsume:
        checkConsume(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::memLoad:
        checkUse(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memStoreOwned:
        checkDef(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memInOut:
        // It is sufficient to just check that we're using the input operation,
        // and if this is the last use of the operation, we should insert a
        // destructor for the value.  checkDef would mark the value as
        // not-live-in, which we don't want.
        checkUse(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memConsume:
        checkConsume(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memMarkDestroyed:
        // The lit.ownership.mark_destroyed op consumes the whole object bit of
        // a value only, but not its fields.  This ensures the sub-fields are
        // destroyed but the full object is not.  It is used in destructors
        // primarily.
        for (auto valueRef :
             valueSet.getValueRefsForAccess(operand, /*isDeref=*/true))
          consumedValues.set(valueRef.endBit - 1);
        break;
      case OperandEffect::memStoreConditional:
        checkDef(operand, op, /*isDeref=*/true, /*needsCheckUse=*/false);
        break;
      }
    }

    // Process any other indirect lifetimes accessed.
    for (auto lifetime : lifetimeEffects)
      checkLifetimeEffect(lifetime, op);
  }
}

/// This is returned when the op is a return or unreachable op.
void DestructorInsertion::checkTerminatorOp(Operation &op) {
  consumedValues.reset();
  if (isa<UnreachableOp>(op))
    return;

  assert((isa<KGEN::ReturnOp, ErrorReturnOp>(op)) && "unknown terminator");
  consumedValues.set(0); // Slot 0 indicates that this block is reachable.

  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    if (isUninitializedAtExit(valueInfo, op))
      continue;

    consumedValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
  }
}
void DestructorInsertion::checkLocalControlFlowOp(Operation &op) {
  if (isa<HLCF::BreakOp, ParamForBreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    consumedValues = *breakSet;
    return;
  }
  if (isa<HLCF::ContinueOp, ParamForContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    consumedValues = *continueSet;
    return;
  }

  // A raise will use the consume set that was seen on entry to the enclosing
  // except block.
  assert(isa<LIT::TryRaiseOp>(op) && "Unknown local control flow op");
  assert(raiseSet && "Not in a 'try'?");
  consumedValues = *raiseSet;
}

void DestructorInsertion::checkAcyclicControlFlowOp(Operation &op) {
  auto controlFlowNode = cast<HLCF::ControlFlowNode>(op);
  SmallVector<SmallVector<int>> succ(op.getNumRegions());
  SmallVector<SmallVector<int>> pred(op.getNumRegions());
  SmallVector<int> entries;

  // Initialize paths.
  SmallVector<HLCF::ControlFlowTarget> targets;
  SmallVector<Attribute> cfnOperands(controlFlowNode->getNumOperands());
  controlFlowNode.getEntryTargets(cfnOperands, targets);
  for (HLCF::ControlFlowTarget target : targets) {
    if (target.index.has_value()) {
      pred[target.index.value()].push_back(ParentPrev);
      entries.push_back(target.index.value());
    }
  }

  // Compute successor list and predecessor list.
  llvm::DenseSet<int> errorRegions;
  llvm::DenseSet<int> throwRegions;
  for (unsigned curr = 0, e = op.getNumRegions(); curr < e; curr++) {
    Region &region = op.getRegion(curr);
    HLCF::ControlFlowTerminator term =
        cast<HLCF::ControlFlowTerminator>(region.front().getTerminator());

    // Terminator branches outside this op.
    if (!term.isParentNode(&op) || isa<UnreachableOp>(term)) {
      if (isa<LIT::ErrorReturnOp>(term))
        errorRegions.insert(curr);

      if (isa<HLCF::ContinueOp, ParamForContinueOp>(term))
        succ[curr].push_back(Continue);

      if (isa<HLCF::BreakOp, ParamForBreakOp>(term))
        succ[curr].push_back(Break);

      if (isa<LIT::TryRaiseOp>(term)) {
        succ[curr].push_back(Raise);
        throwRegions.insert(curr);
      }

      continue;
    }
    SmallVector<HLCF::ControlFlowTarget> termTargets;
    SmallVector<Attribute> operands(term->getNumOperands());
    term.getBranchTargets(operands, termTargets);
    for (HLCF::ControlFlowTarget t : termTargets) {
      if (t.index.has_value()) {
        succ[curr].push_back(t.index.value());
        pred[t.index.value()].push_back(curr);
        continue;
      }
      succ[curr].push_back(ParentPost);
    }
  }

  DenseMap<int, BitVector> consumedValuesInRegion;
  consumedValuesInRegion[ParentPost] = consumedValues;
  if (continueSet)
    consumedValuesInRegion[Continue] = *this->continueSet;
  if (breakSet)
    consumedValuesInRegion[Break] = *this->breakSet;
  if (raiseSet)
    consumedValuesInRegion[Raise] = *this->raiseSet;

  SmallVector<int> empty;
  auto getSuccessors = [&](int curr) -> SmallVector<int> & {
    if (curr < 0 && curr != ParentPrev)
      return empty;
    return curr == ParentPrev ? entries : succ[curr];
  };
  auto getPredecessors = [&](int curr) -> SmallVector<int> & {
    if (curr < 0)
      return empty;
    return pred[curr];
  };

  // Partially order blocks so that a block's successors are processed before
  // it.
  SmallVector<int> sortedBlocks;
  SmallVector<int> worklist;
  for (int entry : entries)
    worklist.push_back(entry);
  while (!worklist.empty()) {
    int curr = worklist.pop_back_val();
    if (std::find(sortedBlocks.begin(), sortedBlocks.end(), curr) !=
        sortedBlocks.end())
      continue;
    sortedBlocks.push_back(curr);
    for (int s : getSuccessors(curr)) {
      if (s < 0)
        continue;
      bool allPredsSeen = true;
      for (int p : getPredecessors(s)) {
        if (std::find(sortedBlocks.begin(), sortedBlocks.end(), p) ==
            sortedBlocks.end()) {
          allPredsSeen = false;
          break;
        }
      }
      if (!allPredsSeen)
        continue;
      worklist.push_back(s);
    }
  }

  // The self value requires special correction in the context of handling
  // conditional self initialization.
  int selfInitIndex = -1;
  for (ValueInfo &v : valueSet.getValueInfos())
    if (v.isIndirect && v.isFullObjectLiveOnEntry)
      selfInitIndex = v.endValueBit - 1;

  // If a success region does not consume a self init value and there is an
  // error region, the unifier will try and insert an illegal destructor.
  llvm::DenseSet<int> errorToValueNoDestruction;

  // Scan all blocks to insert destructors. Insert parentPrev so it's successors
  // are checked for self initialization.
  sortedBlocks.insert(sortedBlocks.begin(), ParentPrev);
  for (auto currPtr = sortedBlocks.rbegin(); currPtr != sortedBlocks.rend();
       currPtr++) {
    int curr = *currPtr;
    BitVector consumedInSomeSucc(consumedValues.size(), false);
    SmallVector<int> &successors = getSuccessors(curr);
    for (unsigned successor : successors) {
      assert(consumedValuesInRegion.contains(successor) &&
             "a successor to current has not been processed, which suggests a "
             "cycle!");
      BitVector &consumptionInSucc = consumedValuesInRegion[successor];
      consumedInSomeSucc |= consumptionInSucc;
    }

    if (curr == ParentPrev)
      continue;
    Region &region = op.getRegion(curr);
    BitVector oldConsumedValues(consumedValues);
    consumedValues = consumedInSomeSucc;
    scanBlock(region.front());
    consumedValuesInRegion[curr] = consumedValues;
    consumedValues = oldConsumedValues;

    bool isError = errorRegions.contains(curr);

    // Correct unification exception case 1: a region that overwrites a
    // subfield. Ignore error case since that overwrite is artificial.
    BitVector &consumedHere = consumedValuesInRegion[curr];
    BitVector consumedInRegion(consumedHere);
    if (consumedInSomeSucc == consumedHere)
      consumedInRegion.reset();
    else if (!successors.empty())
      consumedInRegion.reset(consumedInSomeSucc);
    for (const ValueInfo &v : valueSet.getValueInfos()) {
      if ((int)(v.endValueBit - 1) == selfInitIndex && isError)
        continue;
      // We have identified a full object. Makes sure the subfields are
      // reset so that we don't render a full object indestructible by
      // destroying its subfields.
      if (v.isIndirect && consumedInRegion.test(v.endValueBit - 1)) {
        BitVector destroySubfields(consumedHere);
        destroySubfields.flip();
        destroySubfields.reset(0, v.startValueBit);
        destroySubfields.reset(v.endValueBit, destroySubfields.size());
        if (!destroySubfields.none()) {
          if (!dryRun)
            destroyValuesAtEntry(destroySubfields,
                                 op.getRegion(*currPtr).front(), op.getLoc());
          consumedHere.set(v.startValueBit, v.endValueBit);
        }
      }
    }
  }

  // Unify Destructor paths.
  bool needsUpdate = true;
  int i = 0;
  while (needsUpdate) {
    assert(i++ < 2 && "This should be executed at most twice because elif "
                      "nodes have at most one predecessor.");
    needsUpdate = false;
    for (auto currPtr = sortedBlocks.rbegin(); currPtr != sortedBlocks.rend();
         currPtr++) {
      int curr = *currPtr;
      // for each branch, insert destructors for values that are destroyed in
      // some other branch
      BitVector consumedInSomeSucc(consumedValues.size(), false);
      for (int successor : getSuccessors(curr)) {
        // Ignore the contribution if the successor is unreachable.
        if (!consumedValuesInRegion[successor][0])
          continue;
        consumedInSomeSucc |= consumedValuesInRegion[successor];
      }
      for (int successor : getSuccessors(curr)) {
        // Only self contained successors are corrected.
        if (successor < 0)
          continue;
        BitVector consumedInAltBranch(consumedInSomeSucc);
        consumedInAltBranch ^= consumedValuesInRegion[successor];
        consumedValuesInRegion[successor] = consumedInSomeSucc;
        if (consumedInAltBranch.none())
          continue;
        needsUpdate = true;
        // Do not destroy the self out the error/throw regions.
        if (selfInitIndex > -1 && (errorRegions.contains(successor) ||
                                   throwRegions.contains(successor)))
          consumedInAltBranch.reset(selfInitIndex);
        if (!dryRun)
          destroyValuesAtEntry(consumedInAltBranch,
                               op.getRegion(successor).front(), op.getLoc());
      }
    }
  }

  // All entry paths have unified consumed values; it doesn't matter which we
  // use to update the consumed values.
  consumedValues = consumedValuesInRegion[entries.front()];
}

/// 'if' operations propagate the consume sets into each branch, and use the
/// resulting consume sets to make sure the upward propagated set of consumed
/// values is consistent.
void DestructorInsertion::checkIfLikeOp(Operation &ifElseOp) {
  // Given an 'if' like operation (normal 'if' statement, parameter if, or a
  // throwing call) perform dtor analysis for each side and insert destructors
  // at the top of the blocks to form a common upward-projected consume set.
  assert(ifElseOp.getNumRegions() == 2 && ifElseOp.getRegion(0).hasOneBlock() &&
         ifElseOp.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");
  BitVector thenConsumedValues = consumedValues;
  scanBlock(ifElseOp.getRegion(0).front());
  // Scan 'else' block.
  thenConsumedValues.swap(consumedValues);
  scanBlock(ifElseOp.getRegion(1).front());

  unifyConsumedSets(ifElseOp, ifElseOp.getRegion(1).front(),
                    {thenConsumedValues, ifElseOp.getRegion(0).front()});
}

/// Unify consumed sets across two branches of a conditional operation.  The
/// 'consumedValues' set is considered to be the 'consumed' set at the top of
/// the 'consumedValueBlock' block and 'otherInfo' contains the set from the
/// top of some other region.  Both of them meet at condOp.
void DestructorInsertion::unifyConsumedSets(Operation &condOp,
                                            Block &consumedValueBlock,
                                            BlockConsumeInfo otherBlockInfo) {
  // At this point, 'thenConsumedValues' is the set of upwardly consumed
  // values from the 'then' block and 'consumedValues' is the set of upwardly
  // consumed values from the else branch.  See if they agree already, then
  // there is nothing to do.
  if (consumedValues == otherBlockInfo.consumedValues)
    return;

  // We don't want to perform meets with unreachable code (e.g. from `if False:
  // stuff`: if either of the regions is unreachable, then propagate the other
  // one.  This matters because there is no conservative "missing" set for whole
  // object bits.  We use the sentinel's consume bit to know if anything is
  // consumed.
  if (!otherBlockInfo
           .consumedValues[0]) // If "then" isn't reachable, return "else".
    return;
  if (!consumedValues[0]) { // If "else" isn't reachable, return "then".
    consumedValues = otherBlockInfo.consumedValues;
    return;
  }

  // Given two consume sets, our upward propagated final set will be the
  // union of both sets.
  BitVector upwardConsumeSet = consumedValues;
  upwardConsumeSet |= otherBlockInfo.consumedValues;

  // It is possible that some subfields out of a value that is fully consumed
  // are not demanded.  For example, consider something like:
  //
  //   fn test(cond: Bool):
  //     # Tracked as pair.{a,b,overall}
  //     var pair = Pair(a=String(), b=String())
  //
  //     if cond:            # <- consumes pair.{a,overall}, but not pair.b
  //       pair.b = String() # <- overwrites pair.b so it isn't consumed
  //       pair.use()        # <- consumes pair upwards
  //       return            # <- consumes nothing
  //     else:               # <- consumes nothing
  //       return            # <- consumes nothing
  //
  // In this situation we know that "pair overall" is live into to one side
  // and not live into the other side, that we'll need to destroy the whole
  // thing... so the upward-propagated union needs to demand all of
  // pair.{a,b,overall}.  Computing this allows us to rewrite this into:
  //
  //   fn test(cond: Bool):
  //     # Tracked as pair.{a,b,overall}
  //     var pair = Pair(a=String(), b=String())
  //
  //     if cond:
  //       pair.b.__del__()  # the body doesn't demand pair.b, so destroy it
  //       pair.b = String()
  //       pair.use()
  //       pair.__del__()    # destroyed after pair.use's last use.
  //     else:
  //       pair.__del__()    # block doesn't demand pair at all.
  //       return
  //
  // This only happens when the whole object bit is demanded in one set, but not
  // the other for an entire top-level object.  We know this is the case because
  // any use of a subfield will end up demanding the object as a whole.  If we
  // see this, have the union set demand the whole object so it can be
  // destroyed.
  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    // If the whole-object consume bits agree on both sides, then there is
    // nothing to do.
    if (!valueInfo.isIndirect)
      continue; // Register values have a single bit.

    // If it is missing in one side or the other, then the upward set needs to
    // consume the entire object.
    size_t endBit = valueInfo.endValueBit - 1;
    if (consumedValues[endBit] != otherBlockInfo.consumedValues[endBit])
      upwardConsumeSet.set(valueInfo.startValueBit, valueInfo.endValueBit);
  }

  // If we are in a dryrun, just return the computed union of the two sets.
  if (dryRun) {
    consumedValues = upwardConsumeSet;
    return;
  }

  // Otherwise we have to emit destructors for any non-trivial members to get
  // the branches to line up. If the one branch consumed values that the other
  // branch didn't, then we need to destroy those corresponding values in the
  // other branch.

  // destroyValuesAtEntry will mutate consumedValues, so do the block this set
  // represents first.

  // needToConsumeInElse = upwardConsumeSet & ~consumedValues.
  BitVector needToConsumeInElse = upwardConsumeSet;
  needToConsumeInElse.reset(consumedValues);
  destroyValuesAtEntry(needToConsumeInElse, consumedValueBlock,
                       condOp.getLoc());

  // Next handle the "other" set.

  //    needToConsumeInThen = upwardConsumeSet & ~thenConsumedValues.
  BitVector needToConsumeInThen = upwardConsumeSet;
  needToConsumeInThen.reset(otherBlockInfo.consumedValues);
  destroyValuesAtEntry(needToConsumeInThen, otherBlockInfo.block,
                       condOp.getLoc());

  // Restore consumedValues to the merged set.
  consumedValues = upwardConsumeSet;
}

/// For a loop, we know the consume sets for any break statements, but need
/// to iterate the loop to find the right continue sets to use.
///
/// In terms of form, standard for loops will already have their 'else' block
/// removed (merging the logic into the loop body on the exit) but @parameter
/// 'for' statements still have an explicit 'else' block.
void DestructorInsertion::checkLoopOp(Operation &loopOp) {
  // True if this is a parameter for, false if this is an infinite HLCF::LoopOp.
  bool isParamFor = isa<ParamForOp>(loopOp);

  auto loopBodySets = DestructorInsertion::copy(*this);
  // Any 'break's within the loop will produce the consume set for the
  // statement immediately after the loop.  However, @parameter for statements
  // may have an 'else' block that break statements skip over. Save the exit
  // set for break statements.
  BitVector breakSet(consumedValues);

  // If there is an 'else' on a @parameter for, process it to determine the
  // consume set going into the bottom of the loop.
  if (loopOp.getNumRegions() == 2)
    scanBlock(loopOp.getRegion(1).front());

  // The original set will be what any 'break' statement sees.
  loopBodySets.breakSet = &breakSet;

  // The continueSet is the set of values consumed upwards from the top of the
  // loop and carried over the loop.
  //
  // In the case of an infinite HLCF loop, we start the set with no values to be
  // consumed, and with sentinel slot #0 unset indicating that the continue
  // point isn't reachable.  This will cause the first iteration to propagate
  // values up from the 'break' points to the consume set.
  //
  // In the case of a @parameter for (which terminates implicitly when the
  // iterations are done) we propagate up the consume set at the top of the
  // 'else' block.
  auto continueSet = isParamFor ? consumedValues : BitVector(breakSet.size());
  loopBodySets.continueSet = &continueSet;

  // We need to dry run the body evaluation until we get to a stable
  // continue set.
  loopBodySets.dryRun = true;

  // Iteratively scan the loop body until the continue set converges.
  [[maybe_unused]] unsigned numIters = 0;
  while (true) {
    // Scan the body: any breaks will intersect their live-out set with
    // 'breakSet', and any continues will intersect their live-out set with
    // 'continueSet'.
    loopBodySets.scanBlock(loopOp.getRegion(0).front());

    // If we scanned the body and didn't find any live code, then we know
    // there must not be any break statements in it.  Just consider the
    // continue point reachable for the next iteration.
    if (!loopBodySets.consumedValues[0])
      loopBodySets.consumedValues[0] = true;

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

    // Correct values in the continue set. Any value that is both
    // (a) not consumed in consumedValues (after loop)
    // (b) partially consumed in continueSet (we referenced a subfield of this
    // value in loop) should be fully consumed in the loop.
    for (auto [index, valueInfo] : llvm::enumerate(valueSet.getValueInfos())) {
      if (!valueInfo.isIndirect)
        continue; // Register values only have a single bit.

      // If the whole-value is already considered live, then there is nothing
      // to do.
      if (continueSet.test(valueInfo.endValueBit - 1))
        continue;

      // FIXME: This is checking the break set, which isn't right. We actually
      // want the live in to any blocks that break, not the result of those
      // blocks.  This should be handled by normal 'if' merging.
      ValueRef valueRef(index, valueInfo.startValueBit, valueInfo.endValueBit,
                        valueInfo.isIndirect);
      if (!valueRef.isAllMissing(breakSet))
        continue;

      // If some values are live across the loop then make all of them live
      // across the loop.
      if (!valueRef.isAllMissing(continueSet))
        continueSet.set(valueInfo.startValueBit, valueInfo.endValueBit);
    }

    loopBodySets.scanBlock(loopOp.getRegion(0).front());
  }

  consumedValues = std::move(loopBodySets.consumedValues);
}

void DestructorInsertion::checkTryOp(LIT::TryOp tryOp) {
  // The except block is processed with a copy of the consumed value set
  // from the bottom of the try.  After processing it, we know what the
  // consumed values are for the exception block.
  auto exceptSets = DestructorInsertion::copy(*this);
  exceptSets.raiseSet = raiseSet;

  Region &exceptRegion = tryOp.getExceptRegion();
  exceptSets.scanBlock(exceptRegion.front());
  // The except block initializes its block arguments, so if these are tracked
  // we must mark them as consumed.
  for (Value blockArg : exceptRegion.getArguments()) {
    ValueRef valueRef = valueSet.getDirectValueRef(blockArg, /*isDeref=*/false);
    if (!valueRef)
      continue;
    if (!exceptSets.consumedValues[valueRef.startBit]) {
      // There were no references to the owned arguments, so generate a
      // destructor at beginning of the block.
      mlir::ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
          exceptRegion.getLoc(), &exceptRegion.front());
      destroyValueIfNeeded(blockArg, valueRef, builder,
                           /*opWithUse=*/nullptr);
      valueRef.markBits(consumedValues, false);
    } else {
      valueRef.markBits(exceptSets.consumedValues, false);
    }
  }

  // The normal flow finishes with the else block, process it to see what
  // the input consumedValues set to the else block is.
  scanBlock(tryOp.getElseRegion().front());

  // Ok, finally we process the try body.  Any 'raise's within the try body
  // use the consumed values set on entry to the except block.
  llvm::SaveAndRestore x(raiseSet, &exceptSets.consumedValues);
  scanBlock(tryOp.getTryRegion().front());
}

// When the specified value is consumed by an operation we know it doesn't need
// to be destroyed above this point.
void DestructorInsertion::checkConsume(Value value, Operation &op,
                                       bool isDeref) {
  ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref);
  // Uninitialized variable tracking already rejects consumes of indirect
  // non-trivial values.
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
    if (valueSet.isTrivial(value, isDeref))
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

  if (!dryRun) {
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), &op);
    emitDebugKill(valueRef, builder);
  }
}

/// This operation uses whatever fields are being referenced.  Iff this is the
/// /last/ use of a value, emit a destructor of the overall value.
void DestructorInsertion::checkUse(Value value, Operation &op, bool isDeref) {
  // If needed, emit the destructor immediately after the specified operation.
  auto insertPt = std::next(Block::iterator(&op));
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(), insertPt);
  checkUse(value, builder, /*opWithUse=*/&op, isDeref);
}

/// Check a use of a value.  Iff this is the /last/ use of the value, emit a
/// destructor of the overall value.  The 'opWithUse' value (if present)
/// indicates the operation performing the use.  This enables copy ctor elision,
/// but this is null at the start of block/function for example.
void DestructorInsertion::checkUse(Value value,
                                   mlir::ImplicitLocOpBuilder &builder,
                                   Operation *opWithUse, bool isDeref) {
  // If this is a direct reference to a value, we are tracking it, meaning there
  // are dedicated bits in the consumeValues bitvector that represent the
  // consumption state of this value.
  if (ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref)) {
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
    //
    // This also handles the case of indirect references, resetting to the
    // correct value to destroy.
    // If dryRun, then the upward consume set is unset for values potentially
    // consumed beneath the loop. As a result, we don't know if this is the last
    // reference to whole value. Assuming that it is will result in destruction
    // of that value in a break branch.
    if (value != valueInfo.value &&
        !consumedValues[valueInfo.endValueBit - 1] && !dryRun) {
      value = valueInfo.value;
      valueRef = valueSet.getFullValueRef(valueRef.valueId);
    }

    // Otherwise, it is possible that that ValueRef is live but the overall
    // object will be consumed, this happens in scenarios like:
    //
    //   init(&aggregate)
    //   use(&aggregate.field1)  <<-- We are here.
    //   ... field1 is not consumed here...
    //   aggregate.field1 = newValue  // overwrite field1.
    //   consume(&aggregate)
    //
    // In this case, we need to destroy field1 after this use.
    destroyValueIfNeeded(value, valueRef, builder, /*opWithUse=*/opWithUse);
    return;
  }

  // We are not tracking this value directly, but it is tied to a lifetime
  // declared by a value we do track. If this is the case, check these values
  // for destruction.
  if (isDeref) {
    SmallVector<ValueRef> lifetimeRelatedValues =
        valueSet.getValueRefsForLifetime(
            cast<RefType>(value.getType()).getLifetime());
    for (auto lifetimeRelatedValue : lifetimeRelatedValues) {
      ValueInfo &valueInfo =
          valueSet.getValueInfos()[lifetimeRelatedValue.valueId];
      destroyValueIfNeeded(valueInfo.value, lifetimeRelatedValue, builder,
                           opWithUse);
    }
  }
}

/// This operation defines the specified value.  If the value is dead on
/// arrival, emit a destructor of the value.
void DestructorInsertion::checkDef(Value value, Operation &op, bool isDeref,
                                   bool needsCheckUse) {
  // If there is no use of the value we are defining, emit a dtor after the op.
  // This happens when we have things like:
  //
  //   init(&aggregate)
  //   ...
  //   aggregate.field1 = newValue  <<-- we are here
  if (needsCheckUse)
    checkUse(value, op, isDeref);

  // This call defines the result, so anything above it is either dead or
  // needs a destructor if live.  If this is a direct reference, we mark the
  // target as being consumed.
  if (ValueRef direct = valueSet.getDirectValueRef(value, isDeref)) {
    // FIXME(#579): Rework Error handling in the Compiler so we can remove error
    // handling.
    if (isa<OwnershipMarkInitializedOp>(op)) {
      ValueInfo valueInfo = valueSet.getValueInfo(direct.valueId);
      bool isUninitInErrorBranch =
          valueInfo.endInitState == LifetimeTrackable::EndsUninit ||
          valueInfo.endInitState == LifetimeTrackable::InitOnNormal;
      Operation *term = op.getParentRegion()->front().getTerminator();
      bool isError = isa<ErrorReturnOp, TryRaiseOp>(term);
      // If is initialized in error branch, avoid generating destruction in
      // success branch. Destruction of overwritten memory will be handled at
      // callsite.
      if (!isError && !isUninitInErrorBranch)
        return;
    }

    if (!dryRun && value.getDefiningOp<VarDeclOp>()) {
      mlir::ImplicitLocOpBuilder builder(op.getLoc(), &op);
      emitDebugInit(value, direct, builder);
    }

    direct.markBits(consumedValues, false);
    return;
  }

  // Otherwise, we need to direct-emit a destructor call of the reference
  // itself since this operation will overwrite the value and we can't model it
  // in a field sensitive way.  The uninitialized checker verified that the
  // value is guaranteed live-in when nontrivial and indirect.
  if (!valueSet.isTrivial(value, isDeref) && !dryRun) {
    // Destructor call goes ahead of the mutation.
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), &op);
    emitDestructorCallAt(value, isDeref, builder, &op);
  }
}

/// Check any unstructured lifetimes that are accessed by the operation.
void DestructorInsertion::checkLifetimeEffect(TypedAttr lifetime,
                                              Operation &op) {
  // For destructor insertion, we don't care if this is a read or write.
  // If needed, emit the destructor immediately after the specified operation.
  auto insertPt = std::next(Block::iterator(&op));
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(), insertPt);

  SmallVector<ValueRef> accesses = valueSet.getValueRefsForLifetime(lifetime);
  for (auto access : accesses) {
    // Iff this is the /last/ use of the value, emit a dtor for the value.
    ValueInfo &valueInfo = valueSet.getValueInfos()[access.valueId];
    if (valueInfo.hasErrorDiagnosed)
      continue;

    destroyValueIfNeeded(valueInfo.value, access, builder, /*opWithUse=*/&op);
  }
}

/// If the specified valueRef corresponds to a trivial value or subfield, clear
/// the bits associated with it in 'bits'.  This is recursive, because valueRef
/// may refer to a subfield of the overall value.
static void clearTrivialFields(ValueRef valueRef, Type valueType,
                               BitVector &bits, ValueSet &valueSet) {
  // If all the bits are already clear, we're done.
  if (valueRef.isAllMissing(bits))
    return;

  // If this value is trivial then clear the bits and we're done!
  if (valueSet.isTrivial(valueType, /*isIndirect=*/false)) {
    valueRef.markBits(bits, false);
    return;
  }

  DeclRefType valueDRType = dyn_cast<DeclRefType>(valueType);
  if (!valueDRType) // Trait values are not trivial.
    return;

  // Otherwise, this may be a subfield of an overall value.  Zoom in to see if
  // valueRef is referring to a trivial subfield of the overall object.
  unsigned nextBit = 0;
  for (auto field : valueSet.typeDeclInfo.getStructDeclForType(valueDRType)
                        .getFieldDecls()) {
    unsigned numBits =
        valueSet.typeDeclInfo.getNumFieldsInType(field.getType());
    // If this field has consumed bits, and if has trivial type, force it
    // back to being non-consumed.  This can allow the proper correctness
    // check to work and make the error diagnostic more accurate.
    ValueRef subFieldBits = valueRef.getSubfield(nextBit, numBits);
    clearTrivialFields(subFieldBits, field.getReboundType(valueDRType), bits,
                       valueSet);
    nextBit += numBits;
  }
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
  // If a generic type or trivial, then emit a destructor call (or nothing).
  auto valueType = dyn_cast<DeclRefType>(valueRef.getValueType(value));
  if (!valueType) {
    emitDebugKillAndDestructorCallAt(value, valueRef, builder, opWithUse);
    valueRef.markBits(consumedValues, true);
    return;
  }

  // If the entire value needs to be destroyed, then emit a destructor for the
  // whole value.
  if (!consumedValues.test(valueRef.endBit - 1)) {
    // Trivial types don't have __del__ methods and can't be tracked, so if this
    // is referring to one of them, make sure to clear the bits so we don't
    // think they need to be destroyed.
    clearTrivialFields(valueRef, valueType, consumedValues, valueSet);

    // If a field of a value we must destroy is already destroyed, then we have
    // an error, because we cannot run the destructor on the whole object if one
    // of the fields is missing.
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
              "overall value from being destroyed";
      valueRef.markBits(consumedValues, false);
    }

    // Ok, everything looks good - actually emit the dtor call here.
    emitDebugKillAndDestructorCallAt(value, valueRef, builder, opWithUse);
    valueRef.markBits(consumedValues, true);
    return;
  }

  // Otherwise, we must have an indirect value where some fields are present and
  // some are missing.  Recursively walk the type and destroy just the fields
  // that are missing.
  LIT::StructDeclOp structDecl =
      valueSet.typeDeclInfo.getStructDeclForType(valueType);

  // Initialize an evaluator so that we can resolve the field types.
  ParameterEvaluator evaluator;
  for (auto [decl, value] :
       llvm::zip(structDecl.getParams(), valueType.getParamValues()))
    evaluator.setParameterValue(decl, value);

  unsigned nextBit = 0;
  for (auto field : structDecl.getFieldDecls()) {
    Operation *fieldVal;
    if (!valueRef.isIndirect)
      fieldVal = builder.create<LIT::StructExtractOp>(value, field);
    else
      fieldVal = builder.create<RefStructGEROp>(value, field);

    unsigned numBits = valueSet.typeDeclInfo.getNumFieldsInType(
        evaluator.getReboundType(field.getType()));
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
  assert((isa<PointerType, RefType>(p2.getType())));
  // If the value is an integer or other random thing, then it can't point to
  // anything.
  if (!isa<PointerType, RefType>(p1.getType()))
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
// "tmp" and where "src" gets mutated before the use of "tmp", e.g.:
//
//    %tmp = lit.var.decl "anonymous"
//    kgen.call __copyinit__(%tmp, %src)  <<== Last use of %src
// ** kgen.call __del__(%src)   <<== Thinking about inserting this.
//    kgen.call __init__(%src)  <<== Could reinitialize %src before use of %tmp!
//    use(%tmp)
//    use(%src)
//
// Doing this right requires non-trivial liveness analysis which should
// itself be part of a standalone SSA pass post-inlining.  For now we'll
// just catch the most obvious local cases to clean up the IR and provide a
// "guaranteed" optimization.
static bool canEntirelyElideMemoryTemporary(LIT::CallOp copyInitCall,
                                            VarDeclOp tmpDecl) {
  assert(copyInitCall.getOperand(0) == tmpDecl &&
         "the vardecl is known to be directly assigned");
  Block *tmpBlock = tmpDecl->getBlock();
  if (copyInitCall->getBlock() != tmpBlock)
    return false;

  // Find all users of "tmp".
  SmallPtrSet<Operation *, 3> userOfTmp;
  // Worklist of projections of the tmp VarDecl we need to check.
  SmallVector<Value> valuesToCheck;
  valuesToCheck.push_back(tmpDecl);

  while (!valuesToCheck.empty()) {
    Value checkVal = valuesToCheck.pop_back_val();

    for (OpOperand &operand : checkVal.getUses()) {
      Operation *user = operand.getOwner();

      if (user->getBlock() != tmpBlock)
        return false; // We don't handle control flow.

      // If we see a lit.ref.immut or rebind of the lifetime, check all its uses
      // as well.
      if (isa<RefImmutOp, RebindOp>(user)) {
        valuesToCheck.push_back(user->getResult(0));
        continue;
      }

      // Ignore the copyinit of tmp.
      if (user == copyInitCall)
        continue;

      // Otherwise, the only sort of user we can support is a call that consumes
      // the value.
      // NOTE: This could be extended in the future to be more powerful, e.g. to
      // support patterns like:
      //    %tmp = lit.var.decl "anonymous"
      //    kgen.call __copyinit__(%tmp, %src)  <<== Last use of %src
      // ** kgen.call __del__(%src)   <<== Thinking about inserting this.
      //    use(%tmp)
      //    consume(%tmp)
      // It isn't clear why we're limiting this?
      auto callUser = dyn_cast<LIT::CallOp>(user);
      if (!callUser)
        return false; // Unknown user.

      // The argument convention for the callee must be consuming, not
      // initializing or anything else.
      auto convention =
          callUser.getCalleeType().getArgConvention(operand.getOperandNumber());
      if (convention != ArgConvention::OwnedInMem)
        return false;
      userOfTmp.insert(callUser);
    }
  }

  assert(!userOfTmp.empty() && "tmp should at least be destroyed");

  // Okay, we only see users of the 'tmp' decl that we can understand.  Do a
  // lexical scan to make sure there is nothing between the initialization of
  // the tmp and the use of the tmp that might re-use the source.
  Value srcPointer = copyInitCall.getOperand(1);
  for (auto it = ++Block::iterator(copyInitCall), e = tmpBlock->end();; ++it) {
    // If we ran off the end of the block but we didn't see the users, then the
    // copyinit doesn't dominate this use, something weird is going on, bail
    // out.
    if (it == e)
      return false;

    // Scan all the operands to see if any of them are related to %src. We
    // disallow regions because we don't recurse into them.
    if (it->getNumRegions() || llvm::any_of(it->getOperands(), [&](Value v) {
          return v && mightPointTo(v, srcPointer);
        }))
      return false;

    // If this operation is a known user of tmp, then we might be done scanning.
    if (userOfTmp.erase(&*it)) {
      if (userOfTmp.empty())
        return true;
    }
    // Otherwise, keep looking through the block until we see all the users.
  }
  return true;
}

/// If this is a lit.ref.immut op that removes mutability, look through it.
static Value stripImmutCast(Value value) {
  if (auto cast = value.getDefiningOp<RefImmutOp>())
    return cast.getOperand();
  return value;
}

/// Given a value of reference type, this checks to see if it is immutable, and
/// casts it back to a mutable reference.  This isn't a generally safe operation
/// from a type system perspective, so should only be used for things like
/// destructor insertion that happen after borrow checking.
static Value getMutableRefForPossiblyImmutValue(Value value,
                                                ImplicitLocOpBuilder &builder) {
  value = stripImmutCast(value);

  // Check to see if the reference is already mutable.
  auto destType = cast<RefType>(value.getType()).getWithMutability(true);
  if (value.getType() == destType)
    return value;

  return builder.create<RebindOp>(destType, value);
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
  LIT::FuncOp callee =
      valueSet.typeDeclInfo.getFuncForSymbol(copyInitCall.getDirectCallee());
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
      if (auto load = srcValue.getDefiningOp<LIT::RefLoadOp>()) {
        if (load.getOperand() == value)
          isOk = true;
      }
      if (!isOk)
        return failure();
    }

    // Transform into:
    //   kgen.call user(%value)
    copyInitCall.getResult(0).replaceAllUsesWith(srcValue);

    // We'll delete the copyInit but don't want to invalidate iterators so do it
    // later.  Remove the operand uses so we don't see them in later def-use
    // scans, and to make it more obvious when reading IR dumps that these will
    // be gone.
    copyInitCall->dropAllReferences();
    opsToRemove.push_back(copyInitCall);
    return success();
  }

  // Otherwise handle memory passable copies like:
  //   %tmp = lit.var.decl "anonymous"
  //   kgen.call __copyinit__(%tmp, %src)
  //   kgen.call __del__(%src)   <<= Thinking about inserting this.
  //   kgen.call user(%tmp)      <<= Consuming call.
  if (callee.getSpecialFunctionKind() != SpecialFunctionKind::kCopyInit)
    return failure();

  // Register passable types will pass the 'existing' value in a register copies
  // If we have:
  //   %tmp = lit.var.decl "anonymous"
  //   %srcReg = lit.ref.load %src
  //   lit.call __copyinit__(%tmp, %srcReg)
  //   ==> destroy %src
  // Then we can locally optimize this into:
  //   %tmp = lit.var.decl "anonymous"
  //   %srcReg = lit.ref.load %src
  //   lit.ref.store %srcReg -> %tmp
  // And if the only other operation using '%tmp' is a lit.load.consume:
  //   %tmp = lit.var.decl "anonymous"
  //   %srcReg = lit.ref.load %src
  //   lit.call __copyinit__(%tmp, %srcReg)
  //   ==> destroy %src
  //   ...
  //   %xyz = lit.load.consume %tmp
  // Then we can optimize this into:
  //   %srcReg = lit.ref.load %src
  //   ...
  //   %xyz = %srcReg
  if (auto loadOp = copyInitCall.getOperand(1).getDefiningOp<RefLoadOp>()) {
    if (loadOp.getOperand() == value) {
      Value copyInitDest = copyInitCall.getOperand(0);

      // We're definitely removing the copyinit.
      copyInitCall->dropAllReferences();
      opsToRemove.push_back(copyInitCall);

      // If the operation right after the call is a consuming load from a
      // varDecl, then we can squash the vardecl and the consuming load and
      // avoid emitting a store, tidying things right up.
      if (auto varDecl = copyInitDest.getDefiningOp<VarDeclOp>();
          varDecl && copyInitDest.hasOneUse()) {
        if (auto loadConsume =
                dyn_cast<LoadConsumeOp>(*copyInitDest.user_begin())) {

          // The loadConsume is dead and can be removed.
          loadConsume.getResult().replaceAllUsesWith(loadOp);
          // We know the bottom-up scan won't revisit it, so directly remove.
          loadConsume.erase();

          // The lit.var.decl had one use before so it is now dead, we can
          // remove it as well.
          opsToRemove.push_back(varDecl);
          return success();
        }
      }

      // Otherwise we need to insert a store.  Put the store after the call so
      // it isn't reprocessed by destructor insertion.
      Operation *opAfterCall = &*++Block::iterator(copyInitCall);
      ImplicitLocOpBuilder builder(copyInitCall.getLoc(), opAfterCall);
      builder.create<RefStoreOp>(loadOp, copyInitDest);
      return success();
    }
  }

  // For memory types, make sure we're destroying the whole value, not a
  // subvalue.
  if (copyInitCall.getOperand(1) != value) {
    // The value being destroyed may be a mutable source, and the source of the
    // copy is (by definition) immutable.
    if (stripImmutCast(copyInitCall.getOperand(1)) == value)
      value = copyInitCall.getOperand(1);
    else
      return failure();
  }

  ImplicitLocOpBuilder builder(copyInitCall.getLoc(), copyInitCall);

  // We prefer to completely delete the copy if it is into a temporary location
  // that we can forward.
  //
  // Note: we currently delete explicitly declared temporaries, not just
  // implicit ones.  This is a policy decision, and we should look into
  // the impact on debug information, but generally one wouldn't want debug
  // information to block optimizations.
  if (VarDeclOp tmpDecl =
          copyInitCall.getOperand(0).getDefiningOp<VarDeclOp>()) {
    if (canEntirelyElideMemoryTemporary(copyInitCall, tmpDecl)) {
      // Insert a declaration of the lifetime for the tmp we're eliding, we know
      // that VarDeclOp's always declare a unique lifetime.
      auto refType = cast<RefType>(tmpDecl.getType());
      auto param = cast<ParamDeclRefAttr>(refType.getLifetime());

      // The old reference type used a novel lifetime.  We need to declare it,
      // and coerce back to it.
      builder.create<ParamDeclareOp>(ParamDeclAttr::get(param),
                                     LifetimeAttr::get(param.getType()));
      auto refCasted = builder.create<RebindOp>(tmpDecl.getType(),
                                                copyInitCall.getOperand(1));

      tmpDecl.getResult().replaceAllUsesWith(refCasted);

      // We'll delete the copyInit but don't want to invalidate iterators so do
      // later.  Remove the operand uses so we don't see them in later def-use
      // scans, and to make it more obvious when reading IR dumps that these
      // will be gone.
      copyInitCall->dropAllReferences();
      opsToRemove.push_back(copyInitCall);
      opsToRemove.push_back(tmpDecl);
      return success();
    }
  }

  // Otherwise, try to promote to a __moveinit__ call if present.
  SymbolConstantAttr moveCtor =
      valueSet.typeDeclInfo.getMoveInitForType(destroyedType);
  if (!moveCtor)
    return failure();

  // moveCtor must have __moveinit__(inout self, owned: Self) type.
  auto refValue = cast<RefType>(value.getType());
#ifndef NDEBUG
  auto moveSig = cast<SignatureType>(moveCtor.getType());
  assert(moveSig.getNumArguments() == 2);
  assert(moveSig.getArgConvention(0) == ArgConvention::InitSelf);
  assert(moveSig.getArgConvention(1) == ArgConvention::OwnedInMem);
  auto valueEltType = refValue.getElementType();
  auto moveArgs = moveSig.getArguments();
  auto moveValue1Ref = cast<RefType>(moveArgs[1]);
  // refValue is immutable here because it was passed to a copy.
  assert(cast<RefType>(moveArgs[0]).getElementType() == valueEltType &&
         moveValue1Ref.getElementType() == valueEltType &&
         moveValue1Ref.isMutableKnown(true) && refValue.isMutableKnown(false));

  auto destType = cast<RefType>(copyInitCall.getOperand(0).getType());
  assert(destType.getElementType() == refValue.getElementType());
#endif

  // We know that the input is mutable (otherwise it wouldn't be tracked for
  // destruction), get the reference to a mutable type.
  value = getMutableRefForPossiblyImmutValue(value, builder);
  refValue = cast<RefType>(value.getType());

  // Switch the source operand, and update the lifetime associated with it.
  copyInitCall.setOperand(1, value);
  copyInitCall.setImplicitLifetimes(
      {copyInitCall.getImplicitLifetimes()[0], refValue.getLifetime()});

  // Transform the copy into a move.
  copyInitCall.setCalleeAttr(moveCtor);
  // Since we changed the copy to a __moveinit__, we don't need a dtor call.
  return success();
}

/// Emit one destructor call for one entire value or field.  This should only be
/// called by destroyValueIfNeeded.
///
/// The 'opWithUse' value, if present, is the operation using the overall value
/// being destroyed.  This allows us to perform copy ctor+temp elision.
void DestructorInsertion::emitDestructorCallAt(Value value, bool isIndirect,
                                               ImplicitLocOpBuilder &builder,
                                               Operation *opWithUse) {
  assert(!dryRun && "this inserts!");

  Type destroyedType =
      ValueRef::getDereferencedType(value.getType(), isIndirect);
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
  assert(signature.getNumArguments() == 1 && "dtor should have one operand");

  // We may have a @register_passable value indirect (e.g. because it is in a
  // var).  If so, it needs to be loaded to invoke the destructor.
  Value valueToDestroy = value;
  if (auto ref = dyn_cast<RefType>(valueToDestroy.getType()))
    if (signature.getArguments()[0] == ref.getElementType())
      valueToDestroy = builder.create<RefLoadOp>(valueToDestroy);

  // If the dtor takes a reference, then this the dtor for a memory type.  Bind
  // the implicit lifetime of __del__'s self to the lifetime of the reference we
  // have.
  SmallVector<TypedAttr> implicitLifetimes;
  if (auto delSelfTy = dyn_cast<RefType>(signature.getArguments()[0])) {
    valueToDestroy =
        getMutableRefForPossiblyImmutValue(valueToDestroy, builder);
    auto argRef = cast<RefType>(valueToDestroy.getType());
    assert(delSelfTy.getElementType() == argRef.getElementType());
    implicitLifetimes.push_back(argRef.getLifetime());

    // Verify that the address space of the reference matches.  The __del__
    // method will have address space zero.  Attempts to delete other things
    // should not explode the compiler.
    if (delSelfTy.getAddressSpace() != argRef.getAddressSpace()) {
      mlir::emitError(builder.getLoc())
          << "cannot destroy value in non-default address space";
      return;
    }

  } else {
    assert(signature.getArguments()[0] == valueToDestroy.getType());
  }

  // Emit the call to the destructor.
  builder.create<LIT::CallOp>(signature.getResults()[0], dtor,
                              implicitLifetimes, valueToDestroy);
}

void DestructorInsertion::emitDebugInit(Value value, ValueRef valueRef,
                                        mlir::ImplicitLocOpBuilder &builder) {
  assert(!dryRun && "shouldn't be called in a dry run");
  ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
  // Insert debug value if full value is initialized.
  if (info.debugVariable && valueRef.startBit == info.startValueBit &&
      valueRef.endBit == info.endValueBit) {
    // The IR type needs to be deref'ed to get the source type. Encode the IR
    // type as a pointer type.
    auto diPointerType = DebugInfo::DITargetIndependentPointerType::get(
        info.debugVariable.getType());
    auto newIrValue = DebugInfo::DIIRValueExprAttr::get(diPointerType);
    auto conversion = DebugInfo::DIDerefExprAttr::get(newIrValue);
    builder.create<DebugInfo::ValueOp>(value, info.debugVariable, conversion);
  }
}

void DestructorInsertion::emitDebugKill(ValueRef valueRef,
                                        mlir::ImplicitLocOpBuilder &builder) {
  assert(!dryRun && "shouldn't be called in a dry run");
  // Insert end-of-life debug value if full value is destroyed.
  // TODO(#34115): Emit fragment end-of-life for partial destruction.
  ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
  if (info.debugVariable && valueRef.startBit == info.startValueBit &&
      valueRef.endBit == info.endValueBit) {
    builder.create<DebugInfo::KillOp>(info.debugVariable);
  }
}

void DestructorInsertion::emitDebugKillAndDestructorCallAt(
    Value value, ValueRef valueRef, mlir::ImplicitLocOpBuilder &builder,
    Operation *opWithUse) {
  // We are going to emit a destructor for the specified ValueRef, so all none
  // of the things we are about to destroy should already be destroyed.
  assert(valueRef.isAllMissing(consumedValues) &&
         "cannot have partially consumed object");
  emitDebugKill(valueRef, builder);
  emitDestructorCallAt(value, valueRef.isIndirect, builder, opWithUse);
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

  // Check if the local variables of this function need debug info.
  DebugInfo::DISubprogramAttr funcSpAttr = func.getSubprogramScope();
  DebugInfo::DICompileUnitAttr compileUnit =
      funcSpAttr ? funcSpAttr.getCompileUnit() : nullptr;
  const bool genDebugInfo = compileUnit && compileUnit.getEmissionKind() ==
                                               DebugInfo::EmissionKind::Full;

  SmallVector<bool> argShadowed(func.getNumArguments(), false);
  func.getBody()->walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) -> WalkResult {
        // Skip looking at nested functions, they are handled as separate
        // contexts.
        if (isa<LIT::FuncOp>(op))
          return WalkResult::skip();

        // All the ops that define trackable values have a single result.
        if (op->getNumResults() == 1) {
          Value result = op->getResult(0);
          if (auto trackable = LifetimeTrackable(result)) {
            // Generate debug info for VarDecls if needed.
            DebugInfo::DILocalVariableAttr debugVariable;
            if (genDebugInfo) {
              if (auto varDecl = dyn_cast<VarDeclOp>(op)) {
                debugVariable =
                    createDebugVariableForVarDecl(varDecl, funcSpAttr);
                if (varDecl.getArgShadowIndex())
                  argShadowed[*varDecl.getArgShadowIndex()] = true;
              }
            }

            valueSet.addValue(result, trackable, debugVariable);
          }
        }

        // If there are any regions, check the block arguments for arguments.
        for (auto &region : op->getRegions()) {
          for (auto &block : region)
            for (auto arg : block.getArguments())
              if (auto trackable = LifetimeTrackable(arg))
                valueSet.addValue(arg, trackable);
        }

        return WalkResult::advance();
      });

  ArrayRef<PogMetadataAttr> pogList =
      func.getSignature().getArgListAttrs().getPogs();
  OpBuilder debugBuilder = OpBuilder::atBlockBegin(func.getBody());
  for (BlockArgument arg : func.getArguments()) {
    DebugInfo::DILocalVariableAttr debugVariable;
    if (genDebugInfo && !argShadowed[arg.getArgNumber()])
      debugVariable = insertDebugVariableForArg(debugBuilder, func, arg,
                                                pogList, funcSpAttr);
    if (auto trackable = LifetimeTrackable(arg))
      valueSet.addValue(arg, trackable, debugVariable);
  }

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined, emitting diagnostics as we go.
  UninitializedValueScan(valueSet).scanFunction(func);

  // TODO: How do we want to handle captures in closures?  Their uses
  // effectively form the capture list for the closure.  Should this get
  // materialized by LowerSemanticCF before this pass?
  SmallVector<Operation *> opsToRemove;
  DestructorInsertion(valueSet, opsToRemove).scanFunction(func);

  // Remove copy ctors and allocations that have been elided.
  for (Operation *op : opsToRemove)
    op->erase();

  // Return failure if we generated errors for any of the tracked values.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  }));
}
