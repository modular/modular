//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass checks value lifetime invariants, e.g. that variables are defined
// before use. This also inserts destructors for implicitly destroyed values.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/OriginTrackable.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

static constexpr StringRef extraOriginUsesAttrName = ".mojo.extra.origin.uses";

/// Find all the functions and types in the module.
static std::tuple<std::vector<FnOp>, DenseMap<SymbolRefAttr, FnOp>,
                  DenseMap<SymbolRefAttr, LIT::StructDeclOp>,
                  DenseMap<SymbolRefAttr, LIT::TraitDeclOp>>
collectFunctionsAndTypes(Operation *module) {
  std::vector<FnOp> funcList;
  DenseMap<SymbolRefAttr, FnOp> funcMap;
  DenseMap<SymbolRefAttr, LIT::StructDeclOp> structMap;
  DenseMap<SymbolRefAttr, LIT::TraitDeclOp> traitMap;
  module->walk([&](Operation *op) {
    // Collect functions and nested functions.
    if (auto funcOp = dyn_cast<FnOp>(op)) {
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
      /*alignInBits=*/0, sourceType, DebugInfo::DIFlags::Zero);

  return varAttr;
}

/// Inserts a DebugInfo::ValueOp for this block argument if necessary.
/// `funcSpAttr` is the DISubprogramAttr of the surrounding function `func`.
/// Returns the VarInfo of the inserted ValueOp.
static DebugInfo::DILocalVariableAttr
insertDebugVariableForArg(OpBuilder &builder, FnOp func, BlockArgument arg,
                          ArrayRef<PogMetadataAttr> pogList,
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

  DebugInfo::DIExprAttr diExpr =
      DebugInfo::DIIRValueExprAttr::get(arg.getType());

  // If this argument has address, its needs an initial deref.
  ArgConvention convention =
      func.getFuncTypeGenerator().getArgConvention(arg.getArgNumber());
  if (hasAddress(convention)) {
    if (auto argRefType = dyn_cast<RefType>(arg.getType())) {
      diExpr =
          DebugInfo::DIDerefExprAttr::get(diExpr, argRefType.getElementType());
    }
  }

  DebugInfo::DIType sourceType =
      DebugInfo::DIUnresolvedMLIRType::get(diExpr.getType());
  DebugInfo::DIFlags flags = DebugInfo::DIFlags::Zero;
  if (convention == ArgConvention::ByRefError ||
      convention == ArgConvention::ByRefResult)
    flags = DebugInfo::DIFlags::Artificial;

  DebugInfo::DILocalVariableAttr varAttr = DebugInfo::DILocalVariableAttr::get(
      funcSpAttr, name, funcSpAttr.getFile(), fileLoc.getLine(),
      arg.getArgNumber() + 1,
      /*alignInBits=*/0, sourceType, flags);
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
               DenseMap<SymbolRefAttr, FnOp> &&funcMap,
               DenseMap<SymbolRefAttr, LIT::TraitDeclOp> &&traitMap)
      : structMap(std::move(structMap)), funcMap(std::move(funcMap)),
        traitMap(std::move(traitMap)) {}

  /// Return the total number of flattened fields in the specified type.
  unsigned getNumFieldsInType(Type type) const;

  /// Return the start bit for a field with the specified name in the specified
  /// type.
  unsigned getFieldIndex(LIT::StructType type, StringAttr fieldName) const;
  int getFieldIndexOrInvalid(LIT::StructType type, StringAttr fieldName) const;

  /// Given a subfield bit index that indicates a stored field in the specified
  /// type, return the StructFieldOp of the accessed field, the first bit
  /// number covered by the subfield, and the total bits covered by the field.
  std::tuple<StructFieldOp, unsigned, unsigned>
  getFieldContaining(LIT::StructType type, unsigned bitIndex) const;

  /// Return the struct decl for the specified StructType.
  LIT::StructDeclOp getStructDeclForType(LIT::StructType type) const {
    auto it = structMap.find(type.getSymbol());
    assert(it != structMap.end() && "reference to struct that wasn't declared");
    return it->second;
  }

  /// Return true if the specified type is RegisterPassableTrivial - no copy,
  /// move, or destructor members.
  bool isRegisterPassableTrivial(Type type) const;

  /// Given the RValue type for a value that needs to be destroyed, return the
  /// destructor the invoke, or null if there is none.
  TypedAttr getDestructorForType(Type type) const;
  SymbolConstantAttr getMoveInitForType(Type type) const;

  /// If this is a non-destructible/linear type, emit its linear type error
  /// message and return true. Otherwise returns false.
  bool
  emitErrorMsgIfLinearType(Location loc, Type type,
                           std::vector<InFlightDiagnostic> &diagsToEmit) const;

  /// Return the function for a given symbol name if known.
  FnOp getFuncForSymbol(SymbolRefAttr symbolRef) const {
    auto it = funcMap.find(symbolRef);
    return it != funcMap.end() ? it->second : FnOp();
  }

  /// The next anonymous origin number to use in this function.
  size_t nextAnonOriginNumber = 0;

private:
  DenseMap<SymbolRefAttr, StructDeclOp> structMap;
  DenseMap<SymbolRefAttr, FnOp> funcMap;
  DenseMap<SymbolRefAttr, TraitDeclOp> traitMap;

  /// This keeps track of the number of fields in the struct specified by the
  /// (fully flattened) symbol and parameters.
  mutable DenseMap<LIT::StructType, unsigned> numFields;

  /// A map from struct name and field name to index within the struct.  This
  /// isn't the field number, this is the number of recursively flattened
  /// fields until the start of the field.
  mutable DenseMap<std::pair<SymbolRefAttr, StringAttr>, unsigned> fieldIndices;
};

/// Return true if the specified type is RegisterPassableTrivial - no copy,
/// move, or destructor members.
bool TypeDeclInfo::isRegisterPassableTrivial(Type type) const {
  if (auto valueType = dyn_cast<LIT::StructType>(type))
    return getStructDeclForType(valueType).isRegisterPassableTrivial();

  // This is not trivial if it is a reference to a trait value.
  if (auto paramRef = dyn_cast<ParamType>(type)) {
    if (isa<TraitType>(paramRef.getParam().getType()))
      return false;
  }

  // Other values of raw MLIR type are always trivial.
  return true;
}

static SymbolConstantAttr getSpecialMemberForType(
    Type type, const TypeDeclInfo *typeDecls,
    llvm::function_ref<SymbolConstantAttr(StructDeclOp)> getMember) {
  auto valueType = dyn_cast<LIT::StructType>(type);
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
  auto newSig = attr.getType().getSpecializedGenerator(paramValues);
  return SymbolConstantAttr::get(attr.getSymbol(), newSig, paramValues);
}

/// Given the RValue type for a value that needs to be destroyed, return the
/// destructor the invoke, or null if there is none.
TypedAttr TypeDeclInfo::getDestructorForType(Type type) const {
  if (auto generic = dyn_cast<ParamType>(type)) {
    if (auto trait = dyn_cast<TraitType>(generic.getParam().getType())) {
      for (SymbolRefAttr symbol : trait.getSymbols()) {
        FuncTypeGeneratorType dtorSig = TraitDeclOp(traitMap.at(symbol))
                                            .getDtorSig()
                                            .value_or(FuncTypeGeneratorType());
        if (dtorSig) {
          // Bind the *(0,0) parameter to a concrete type we're using in this
          // context.
          TypedAttr selfParam = generic.getParam();
          if (trait.getSymbols().size() > 1) {
            // For trait compositions, upcast the self parameter to the dtor
            // expected type.
            auto expectedSelfType = TraitType::get(symbol);
            selfParam = UpcastAttr::get(expectedSelfType, selfParam,
                                        VTableAttr::get(type.getContext(), {}));
          }
          auto specSig = dtorSig.getSpecializedGenerator({selfParam});
          auto delStr =
              StringAttr::get("__del__", StringType::get(type.getContext()));
          return ParamOperatorAttr::get(POC::GetVTableEntry,
                                        {selfParam, delStr}, specSig);
        }
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

/// If this is a non-destructible/linear type, return the error message to
/// emit if an implicit destructor call is required.
bool TypeDeclInfo::emitErrorMsgIfLinearType(
    Location loc, Type type,
    std::vector<InFlightDiagnostic> &diagsToEmit) const {
  if (auto valueType = dyn_cast<LIT::StructType>(type)) {
    StructDeclOp structDecl = getStructDeclForType(valueType);
    std::optional<StringRef> errorMsg = structDecl.getLinearTypeErrorMsg();
    if (!errorMsg)
      return false;

    auto diag = ::mlir::emitError(loc) << *errorMsg;
    diagsToEmit.push_back(std::move(diag));
    return true;
  }

  if (auto generic = dyn_cast<ParamType>(type)) {
    if (auto trait = dyn_cast<TraitType>(generic.getParam().getType())) {
      InFlightDiagnostic diag = ::mlir::emitError(loc);
      bool hasError = false;
      for (SymbolRefAttr symbol : trait.getSymbols()) {
        TraitDeclOp traitDecl(traitMap.at(symbol));
        std::optional<StringRef> errorMsg = traitDecl.getLinearTypeErrorMsg();
        if (errorMsg) {
          if (hasError)
            diag.attachNote();
          diag << *errorMsg;
          hasError = true;
        }
      }

      diagsToEmit.push_back(std::move(diag));
      return hasError;
    }
  }

  // Otherwise, must be an MLIR type like 'index'.
  return false;
}

/// Return the total number of flattened fields in the specified type.
unsigned TypeDeclInfo::getNumFieldsInType(Type type) const {
  // We currently treat all non-struct types as being a single element, even
  // things like kgen.list containing struct types.
  auto declRef = dyn_cast<LIT::StructType>(type);
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
/// type, or -1 if the field isn't found.
int TypeDeclInfo::getFieldIndexOrInvalid(LIT::StructType type,
                                         StringAttr fieldName) const {
  auto it = fieldIndices.find({type.getSymbol(), fieldName});
  return it == fieldIndices.end() ? -1 : it->second;
}

/// Return the start bit for a field with the specified name in the specified
/// type.
unsigned TypeDeclInfo::getFieldIndex(LIT::StructType type,
                                     StringAttr fieldName) const {
  int idx = getFieldIndexOrInvalid(type, fieldName);
  assert(idx >= 0 && "invalid field name for struct type");
  return unsigned(idx);
}

/// Given a subfield bit index that indicates a stored field in the specified
/// type, return the StructFieldOp of the accessed field, the first bit
/// number covered by the subfield, and the total bits covered by the field.
std::tuple<StructFieldOp, unsigned, unsigned>
TypeDeclInfo::getFieldContaining(LIT::StructType declRef,
                                 unsigned bitIndex) const {
  LIT::StructDeclOp decl = getStructDeclForType(declRef);

  ParameterEvaluator evaluator(decl.getInputParams(), declRef.getParamValues());
  // Scan to find the field that contains this.
  unsigned startFieldIdx = 0;
  for (auto field : decl.getFieldDecls()) {
    // This range check is needed to handle zero-sized fields: they don't
    // contain a field even if they start at the beginning of it.
    Type reboundType = evaluator.getReboundType(field.getType());
    unsigned numSubFields = getNumFieldsInType(reboundType);
    if (startFieldIdx <= bitIndex && startFieldIdx + numSubFields > bitIndex)
      return {field, startFieldIdx, numSubFields};
    startFieldIdx += numSubFields;
  }

  llvm_unreachable("invalid index into struct field numbering");
}

//===----------------------------------------------------------------------===//
// ValueInfo / ValueSet tracking
//===----------------------------------------------------------------------===//

namespace {
struct ValueRef;
struct ValueInfo {
  /// This is the declared value being tracked.  This can be null'd out if the
  /// value is completely removed.
  Value value;

  /// This indicates the (first, end] bitrange in the bit vector corresponding
  /// to this value.
  const unsigned startValueBit, endValueBit;

  /// True if this values starts out uninitialized at the beginning of its
  /// lifetime.
  const bool startsUninit;
  /// Enum indicating whether the value is initalized at function exit.
  const OriginTrackable::ExitInitState endInitState;

  /// True if this value lives in memory, not a @register_passable SSA value.
  const bool isIndirect;

  /// True if this is a byref_result argument for a self argument in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  const bool isFullObjectLiveOnEntry;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;

  /// This is true if the value was ever used.
  mutable bool isEverUsed;

  /// If this value needs to be tracked by debug info, this is the information
  /// about the source variable that created this value. Null otherwise.
  DebugInfo::DILocalVariableAttr debugVariable;

  /// Return true if this value contains the specified bit.
  bool contains(unsigned bitNo) const {
    return startValueBit <= bitNo && bitNo < endValueBit;
  }

  StringAttr getName() const {
    assert(value && "cannot get name of null entry");
    return OriginTrackable(value).name;
  }

  /// Return a ValueRef that covers this whole value.  The caller must provide
  /// the valueId.
  ValueRef getFullValueRef(unsigned valueId) const;
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

  /// Return this ValueRef with the base offset subtracted off. This is useful
  /// when reasoning about a subfield inside another object without knowing the
  /// context.
  ValueRef getWithoutBaseOffset(unsigned offset) const {
    assert(startBit >= offset && "not offset by this base");
    return ValueRef(valueId, startBit - offset, endBit - offset, isIndirect);
  }
  ValueRef getWithBaseOffset(unsigned offset) const {
    return ValueRef(valueId, startBit + offset, endBit + offset, isIndirect);
  }

  /// Return true if this value ref is equal or a superset of the specified one.
  bool contains(ValueRef other) const {
    return startBit <= other.startBit && endBit >= other.endBit;
  }
};

/// Return a ValueRef that covers this whole value.  The caller must provide
/// the valueId.
ValueRef ValueInfo::getFullValueRef(unsigned valueId) const {
  return ValueRef{valueId, startValueBit, endValueBit, isIndirect};
}

/// This tracks the values in a function (including nested functions) that are
/// relevant for ownership - that needs to be tracked for uses without being
/// initialized, or that need a destructor to be run.
///
/// This tracks a /completely field sensitive/ view of the values under
/// consideration, including their nested fields in a flattened representation.
/// This gives us a fully precise view of the individual fields, and allows them
/// to be initialized and consumed in a piecewise way.
struct ValueSet {
  // This allows cached dominance computation within the current function.
  mlir::DominanceInfo domInfo;

  /// This provides information about the types referenced from values, e.g. the
  /// number of fields they have.
  TypeDeclInfo &typeDeclInfo;

  /// This provides efficient lookup for origins buried in MLIR types.
  CachedOriginFinder &originFinder;

  /// Initialize the value set with one entry, so index #0 is always invalid and
  /// can be used as a sentinel, and so a null Value is always treated as
  /// untracked.
  ///
  /// This sentinel is also used by DestructorInsertion as a marker for
  /// "unreachable" code to avoid unnecessary meets.
  ValueSet(TypeDeclInfo &typeDeclInfo, FnOp func,
           CachedOriginFinder &originFinder);

  /// Return the number of values we are tracking.
  MutableArrayRef<ValueInfo> getValueInfos() { return valueInfos; }
  ValueInfo &getValueInfo(size_t idx) { return valueInfos[idx]; }
  const ValueInfo &getValueInfo(size_t idx) const { return valueInfos[idx]; }

  /// Remove a tracked value from the valueset maps, and reset its ValueEntry to
  /// have a null Value.
  void eraseValueInfo(Value value);

  /// Return a reference to the entire value with the specified ID.
  ValueRef getFullValueRef(unsigned valueId) const {
    auto &entry = valueInfos[valueId];
    entry.isEverUsed = true;
    return entry.getFullValueRef(valueId);
  }

  /// Given a origin attribute, return the value ref that defines it, and the
  /// known type of that value.  This can return a null type if we don't have
  /// field sensitive information.
  std::pair<ValueRef, Type> getValueRefAndTypeForOrigin(TypedAttr origin) const;

  /// Look up all the value refs that an access with the specified Value and
  /// dereference bit touch.
  SmallVector<ValueRef> getValueRefsForAccess(Value val, bool isDeref);
  SmallVector<ValueRef> getValueRefsForOrigin(TypedAttr origin);

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
  bool isTrivial(Type type, bool isIndirect) const {
    auto eltType = ValueRef::getDereferencedType(type, isIndirect);
    return typeDeclInfo.isRegisterPassableTrivial(eltType);
  }

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
  FnOp func;
  /// These are all of the value infos, indexed by ID #.
  SmallVector<ValueInfo> valueInfos;
  /// This is a lookup from SSA values to the thing they are referencing.
  DenseMap<Value, unsigned> valueInfoIndex;
  /// This is a mapping of origin attrs to the value index that defines them.
  DenseMap<TypedAttr, unsigned> originToValueIndex;

  /// Add a value to the set that we are tracking.  This includes:
  ///  * the MLIR representation for the value itself
  ///  * whether the value is a by-ref to the underlying logical value
  ///  * The bitrange it covers
  void addValue(Value val, const OriginTrackable &trackable,
                DebugInfo::DILocalVariableAttr debugVariable);
};
} // namespace

/// Initialize the value set with one entry, so index #0 is always invalid and
/// can be used as a sentinel, and so a null Value is always treated as
/// untracked.
///
/// This sentinel is also used by DestructorInsertion as a marker for
/// "unreachable" code to avoid unnecessary meets.
ValueSet::ValueSet(TypeDeclInfo &typeDeclInfo, FnOp func,
                   CachedOriginFinder &originFinder)
    : typeDeclInfo(typeDeclInfo), originFinder(originFinder), func(func) {
  addValue(Value(), OriginTrackable(Value()), DebugInfo::DILocalVariableAttr());

  // Check if the local variables of this function need debug info.
  DebugInfo::DISubprogramAttr funcSpAttr = func.getSubprogramScope();
  DebugInfo::DICompileUnitAttr compileUnit =
      funcSpAttr ? funcSpAttr.getCompileUnit() : nullptr;
  bool genDebugInfo = compileUnit && compileUnit.getEmissionKind() ==
                                         DebugInfo::EmissionKind::Full;

  SmallVector<bool> argShadowed(func.getNumArguments(), false);
  func.getBody()->walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) -> WalkResult {
        // Skip looking at nested functions, they are handled as separate
        // contexts.
        if (isa<FnOp>(op))
          return WalkResult::skip();

        // All the ops that define trackable values have a single result.
        if (op->getNumResults() == 1) {
          Value result = op->getResult(0);
          if (auto trackable = OriginTrackable(result)) {
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

            addValue(result, trackable, debugVariable);
          }
        }

        // If there are any regions, check the block arguments for arguments.
        for (auto &region : op->getRegions()) {
          for (auto &block : region)
            for (auto arg : block.getArguments())
              if (auto trackable = OriginTrackable(arg))
                addValue(arg, trackable, DebugInfo::DILocalVariableAttr());
        }

        return WalkResult::advance();
      });

  ArrayRef<PogMetadataAttr> pogList =
      func.getFuncTypeGenerator().getArgListAttrs().getPogs();
  OpBuilder debugBuilder = OpBuilder::atBlockBegin(func.getBody());
  for (BlockArgument arg : func.getArguments()) {
    DebugInfo::DILocalVariableAttr debugVariable;
    if (genDebugInfo && !argShadowed[arg.getArgNumber()])
      debugVariable = insertDebugVariableForArg(debugBuilder, func, arg,
                                                pogList, funcSpAttr);
    if (auto trackable = OriginTrackable(arg))
      addValue(arg, trackable, debugVariable);
  }
}

/// Add a value to the set that we are tracking.  This includes:
///  * the MLIR representation for the value itself
///  * whether the value is a by-ref to the underlying logical value
///  * The bitrange it covers
void ValueSet::addValue(Value val, const OriginTrackable &trackable,
                        DebugInfo::DILocalVariableAttr debugVariable) {
  // Figure out how many bits to track for this value at the value if mem.
  unsigned numValueBits;
  TypedAttr valueOrigin;
  if (!val) {
    numValueBits = 1; // Nothing to do for the sentinel.
  } else if (trackable.isIndirect) {
    // This should be an assertion, but check softly to help IR clients.
    auto refType = dyn_cast<RefType>(val.getType());
    if (!refType) {
      mlir::emitError(val.getLoc())
          << "INTERNAL ERROR: trackable IR value of type " << val.getType()
          << " should have type '!lit.ref': " << val;
      return;
    }
    Type valType = refType.getElementType();
    numValueBits = typeDeclInfo.getNumFieldsInType(valType);

    // Remember the origin if not unknown.
    if (!isa<AnyOriginAttr>(refType.getOrigin()))
      valueOrigin = refType.getOrigin();
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
  if (valueOrigin)
    originToValueIndex[valueOrigin] = valueInfos.size();

  valueInfos.push_back(ValueInfo{
      val, firstValueBit, firstValueBit + numValueBits, trackable.startsUninit,
      trackable.endInitState, trackable.isIndirect,
      trackable.isFullObjectLiveOnEntry,
      /*hasErrorDiagnosed=*/false, /*isEverUsed=*/false, debugVariable});
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
  if (auto funcOp = dyn_cast<FnOp>(func))
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
    case OriginTrackable::EndsInit:
      break;
    case OriginTrackable::EndsUninit:
      os << " EI";
      break;
    case OriginTrackable::InitOnNormal:
      os << " NR";
      break;
    case OriginTrackable::InitOnError:
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
      if (auto fn = dyn_cast_or_null<FnOp>(bbArg.getOwner()->getParentOp()))
        os << fn.getFuncTypeGenerator().getArgName(bbArg.getArgNumber()) << " ";
    }

    os << info.value << "\n";
  }
  os.flush();
}

/// Remove a tracked value from the valueset maps, and reset its ValueEntry to
/// have a null Value.
void ValueSet::eraseValueInfo(Value value) {
  auto it = valueInfoIndex.find(value);
  assert(it != valueInfoIndex.end() && it->second && "not tracking this value");
  valueInfos[it->second].value = Value();
  valueInfoIndex.erase(it);
}

/// Given a origin attribute, return the value ref that defines it, and the
/// known type of that value.  This can return a null type if we don't have
/// field sensitive information.
std::pair<ValueRef, Type>
ValueSet::getValueRefAndTypeForOrigin(TypedAttr origin) const {
  // The mutability of the origin access doesn't affect what ValueRef is
  // accessed.
  origin = OriginMutCastAttr::strip(origin);

  // If the origin has one or more field specifiers like 'a.x.y.z', find
  // the ValueRef for the base and then refine it.
  if (auto field = dyn_cast<OriginFieldAttr>(origin)) {
    auto [valueRef, type] = getValueRefAndTypeForOrigin(field.getBase());
    // If we don't have field sensitive information then we cannot refine the
    // origin.  This also handles the null valueRef case.
    if (!type)
      return {valueRef, type};

    assert(valueRef.isIndirect && "Cannot field refine SSA value access");
    auto fieldName = field.getField();

    // FIXME: Field accesses can be compressed due to subtyping, and we don't
    // keep track of where this happens in the origin, and we don't keep track
    // of the full struct+symbol name for fields.  *This is a bug*.  Until we
    // decide to fix this, this should work.
    auto containerType = dyn_cast<LIT::StructType>(type);
    if (!containerType)
      return {valueRef, Type()};
    int fieldOffset =
        typeDeclInfo.getFieldIndexOrInvalid(containerType, fieldName);
    if (fieldOffset == -1)
      return {valueRef, Type()};

    // Figure out the declared type of the field.
    auto [fieldDecl, _, numFieldBits] =
        typeDeclInfo.getFieldContaining(containerType, fieldOffset);
    assert(fieldDecl.getNameAttr() == fieldName && "index/name mismatch");

    // Refine the ValueRef and type.
    return {valueRef.getSubfield(fieldOffset, numFieldBits),
            fieldDecl.getType()};
  }

  // Otherwise look up the base origin value.
  auto it = originToValueIndex.find(origin);
  if (it == originToValueIndex.end())
    return {};

  const auto &entry = valueInfos[it->second];
  assert(entry.isIndirect);
  return {entry.getFullValueRef(it->second),
          cast<RefType>(entry.value.getType()).getElementType()};
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
        cast<LIT::StructType>(containerType), structGER.getFieldAttr());
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

/// Look up all the value refs that an access to the specified origin could
/// touch.
SmallVector<ValueRef> ValueSet::getValueRefsForOrigin(TypedAttr origin) {
  SmallVector<ValueRef> result;

  // Look through imm cast and unions to find the underlying attrs.
  processRawOrigin(origin, [&](TypedAttr raw) {
    auto [valueRef, type] = getValueRefAndTypeForOrigin(raw);
    if (valueRef) {
      result.push_back(valueRef);
      valueInfos[valueRef.valueId].isEverUsed = true;
    }
  });

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
  // origin-tracked values, figure out what they are.
  if (isDeref)
    return getValueRefsForOrigin(cast<RefType>(value.getType()).getOrigin());

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

  void scanFunction(FnOp func);
  void scanBlock(Block &body);

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op);
  void checkElIfOp(HLCF::ElifOp op);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  void diagnoseUsageError(ValueRef valueRef, Operation &op, bool isDef);
  void checkUse(Value value, Operation &op, bool isDeref);
  void checkDef(Value value, Operation &op, bool isDeref);
  void checkConsume(Value value, Operation &op, bool isDeref);
  void checkMarkDestroyed(Value value, Operation &op);
  void checkOriginEffect(TypedAttr origin, Operation &op);
  void handleAnyOriginUse(Operation &op, ArrayRef<TypedAttr> definedOrigins);

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

[[maybe_unused]] void UninitializedValueScan::dump() const {
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
    auto declRefType = cast<LIT::StructType>(type);

    auto [fieldDecl, fieldBitOffset, numFieldBits] =
        typeDeclInfo.getFieldContaining(declRefType, firstInvalidOffset);
    firstInvalidOffset -= fieldBitOffset;
    nextValidOffset -= fieldBitOffset;
    type = fieldDecl.getType();
    diag << "." << fieldDecl.getName();
  }

  // Dig into the field to ignore trailing members that we don't care about.
  while (nextValidOffset < typeDeclInfo.getNumFieldsInType(type)) {
    auto declRefType = cast<LIT::StructType>(type);
    auto [fieldDecl, startBit, numBits] =
        typeDeclInfo.getFieldContaining(declRefType, 0);
    type = fieldDecl.getType();
    diag << "." << fieldDecl.getName();
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
  if (valueEntry.getFullValueRef(valueRef.valueId).isAllMissing(bits)) {
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
  valueRef.markBits(liveValues, false);
}

/// The lit.ownership.mark_destroyed op consumes the whole object bit of
/// a value only, but not its fields.  It marks the final aggregate as
/// uninitialized.
void UninitializedValueScan::checkMarkDestroyed(Value value, Operation &op) {
  ValueRef access = valueSet.getDirectValueRef(value, /*isDeref=*/true);
  if (!access) {
    mlir::emitError(op.getLoc(),
                    "can only mark directly tracked values as destroyed");
    return;
  }

  ValueInfo &info = valueSet.getValueInfo(access.valueId);
  if (access != info.getFullValueRef(access.valueId)) {
    if (!info.hasErrorDiagnosed)
      mlir::emitError(op.getLoc(),
                      "can only mark full values as destroyed, not subfields");
    info.hasErrorDiagnosed = true;
    return;
  }

  // Check that the consumed bit is live, otherwise it cannot be destroyed.
  ValueRef fullObjectBit = access.getSubfield(access.getNumBits() - 1, 1);

  // If not, then there is an error which we diagnose.
  if (!fullObjectBit.isAllPresent(liveValues)) {
    diagnoseUsageError(fullObjectBit, op, /*isDef=*/false);
    return;
  }

  // From this point on, the whole value is uninitialized.
  access.markBits(liveValues, false);
}

/// Check any unstructured origins that are accessed by the operation.
void UninitializedValueScan::checkOriginEffect(TypedAttr origin,
                                               Operation &op) {
  // We assume this may mutate the origin unless we know it is read-only.
  bool isMutate = !cast<OriginType>(origin.getType()).isMutableKnown(false);

  SmallVector<ValueRef> accesses = valueSet.getValueRefsForOrigin(origin);
  for (auto access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveValues))
      diagnoseUsageError(access, op, /*isDef=*/isMutate);
  }
}

/// This function is called when an operation uses a #lit.any.origin origin.
/// This happens when the operation accesses through (e.g.) an unbound
/// UnsafePointer.  We don't know what objects may be touched by this access,
/// but we want to ensure (for usability sake) that any origin-tracked values
/// are treated as a use, so they don't get destroyed too early.
///
/// We handle this by learning which things need extension in this function,
/// then attaching an attribute that destructor insertion pass will notice in
/// the second pass.
void UninitializedValueScan::handleAnyOriginUse(
    Operation &op, ArrayRef<TypedAttr> definedOrigins) {
  // Turn the list of origins (which might include unions, mutcasts, etc) into
  // the raw underlying origins of values.
  SmallPtrSet<Attribute, 8> definedOriginSet;
  for (auto elt : definedOrigins) {
    // Look through imm cast and unions to find the underlying attrs.
    processRawOrigin(elt, [&](TypedAttr raw) {
      // Ignore field sensitivity of the use: if we have a def of a subfield of
      // the value then we treat it as defining the value.
      while (auto field = dyn_cast<OriginFieldAttr>(raw))
        raw = field.getBase();
      definedOriginSet.insert(raw);
    });
  }

  // Collect a set of value ID's that might be accessed, evaluating each one.
  SmallVector<int32_t> valueIdsToExtend;

  for (unsigned i = 0, e = valueSet.getValueInfos().size(); i != e; ++i) {
    auto &valueInfo = valueSet.getValueInfo(i);
    // Don't mess with things that are in SSA registers - they aren't
    // addressable with a origin.
    if (!valueInfo.value || !valueInfo.isIndirect)
      continue;

    // Can't be a use if the value isn't fully alive here.
    if (!valueSet.getFullValueRef(i).isAllPresent(liveValues))
      continue;

    // Check to see if the operation directly initializes this origin
    // (e.g. by initializing it). If so, we don't want to treat this as a
    // generalized use.
    auto valueOrigin = cast<RefType>(valueInfo.value.getType()).getOrigin();
    if (definedOriginSet.count(valueOrigin))
      continue;

    // Check to see if the value is dominated by this op.  It is possible for
    // values to be fully live that are not reachable, e.g.:
    //
    //     if cond:
    //        var thing = ...
    //        use(thing)
    //     else:
    //        return
    //     # Thing is fully initialized here but doesn't dominate.
    //     use_any_origin(..)
    //
    if (!valueSet.domInfo.properlyDominates(valueInfo.value, &op))
      continue;

    // This value might be accessed, so we want to extend its origin if
    // necessary.
    valueIdsToExtend.push_back(i);
  }

  if (valueIdsToExtend.empty())
    return;
  op.setAttr(extraOriginUsesAttrName,
             mlir::DenseI32ArrayAttr::get(op.getContext(), valueIdsToExtend));
}

void UninitializedValueScan::scanFunction(FnOp func) {
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
  SmallVector<TypedAttr> originEffects;
  SmallVector<TypedAttr> definedOrigins;
  for (Operation &op : block) {
    operandEffects.clear();
    resultEffects.clear();
    originEffects.clear();
    definedOrigins.clear();
    auto overall = getOperationEffects(op, operandEffects, resultEffects,
                                       originEffects, valueSet.originFinder);
    /// If the operation is unknown, ignore it.
    if (overall == OverallOpValueEffect::unknownOp) {
      // NOTE: Can log here when extending things.
      // op.dump();
      continue;
    }

    bool hasAnyOrigin = false;

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
        hasAnyOrigin |=
            isa<AnyOriginAttr>(cast<RefType>(operand.getType()).getOrigin());
        checkUse(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memStoreOwned:
        checkDef(operand, op, /*isDeref=*/true);
        definedOrigins.push_back(cast<RefType>(operand.getType()).getOrigin());
        break;
      case OperandEffect::memMut:
        hasAnyOrigin |=
            isa<AnyOriginAttr>(cast<RefType>(operand.getType()).getOrigin());
        checkUse(operand, op, /*isDeref=*/true);
        checkDef(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memConsume:
        hasAnyOrigin |=
            isa<AnyOriginAttr>(cast<RefType>(operand.getType()).getOrigin());
        checkConsume(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memMarkDestroyed:
        // Mark destroyed doesn't do general origin access.
        checkMarkDestroyed(operand, op);
        break;
      }
    }

    assert(resultEffects.size() == op.getNumResults() &&
           "getOperationEffects returned wrong # effects");
    for (auto [result, effect] : llvm::zip(op.getResults(), resultEffects)) {
#ifndef NDEBUG
      OriginTrackable trackable(result);
      // Perform some general sanity checking of the OriginTrackable
      // implementation.

      // Since this is an op result, the live in/out behavior must match each
      // other: if this weren't true, then control flow paths that didn't cross
      // the op could never be satisfied.
      bool endsUninit = false;
      if (trackable) {
        assert((trackable.endInitState == OriginTrackable::EndsInit ||
                trackable.endInitState == OriginTrackable::EndsUninit) &&
               "invalid end init state for an op result");
        endsUninit = trackable.endInitState == OriginTrackable::EndsUninit;
        assert(trackable.startsUninit == endsUninit &&
               "op results must have same live in/out behavior");
      }
#endif

      switch (effect) {
      case ResultEffect::ignore:
        assert(!trackable && "Origin trackable and CheckLifetimes disagree");
        continue;
      case ResultEffect::regDefine:
        assert(trackable && !trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        checkDef(result, op, /*isDeref=*/false);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true);
        break;
      case ResultEffect::memDefineUninitToUninit:
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToUninit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true);
        definedOrigins.push_back(cast<RefType>(result.getType()).getOrigin());
        break;
      }
    }

    // Process any indirect origins accessed.
    for (auto origin : originEffects) {
      checkOriginEffect(origin, op);
      hasAnyOrigin |= isa<AnyOriginAttr>(origin);
    }

    // If the operation used a #lit.any.origin value, then we treat it as an
    // implicit use of all tracked values.  This ensures that the values are
    // not destroyed too early.
    if (hasAnyOrigin)
      handleAnyOriginUse(op, definedOrigins);

    // Finally, handle any other special per-operation behavior.
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
    case OverallOpValueEffect::elifOp:
      checkElIfOp(cast<HLCF::ElifOp>(op));
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
  if (valueInfo.endInitState == OriginTrackable::EndsUninit)
    return true;

  if (valueInfo.endInitState == OriginTrackable::InitOnNormal)
    return isa<ErrorReturnOp>(exit);

  if (valueInfo.endInitState == OriginTrackable::InitOnError)
    return isa<KGEN::ReturnOp>(exit);
  return false;
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

/// This is HLCF::IfOp or ParamIfOp, which are all if-like.
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

// This is used for the HLCF::ElifOp.
void UninitializedValueScan::checkElIfOp(HLCF::ElifOp op) {
  // ElIf contains pairs of regions in the elifRegions list, which correspond
  // to a 'condition' and a 'if true' block for each condition.  The live-out
  // set is the intersection of all of the live-out sets for each condition.
  MutableArrayRef<Region> ifRegions = op.getElifRegions();
  assert((ifRegions.size() % 2) == 0 && "Must have pairs of regions");

  // The ultimate live-out set is the intersection of each of the "then" blocks,
  // along with the live-out set of the ultimate else.  Start with everything
  // and wittle it down from there.
  BitVector thenLiveOutValues(liveValues.size(), 1);
  BitVector scratchSet;

  for (size_t nextElIfRegion = 0, e = ifRegions.size(); nextElIfRegion != e;
       nextElIfRegion += 2) {
    // Check the next condition accumulating into liveValues.
    scanBlock(ifRegions[nextElIfRegion].front());
    // Save the live set after the condition but before the 'then' block.
    scratchSet = liveValues;

    // Scan the "then" block for this condition, the result is the exit set for
    // this case.
    scanBlock(ifRegions[nextElIfRegion + 1].front());
    thenLiveOutValues &= liveValues;

    // Restore the live-in set to the set of things before the 'then' block.
    std::swap(liveValues, scratchSet);
  }

  // After each of the cases has been evaluated, check the 'else' block.
  scanBlock(op.getElseRegion().front());

  // The live out set of the whole 'elif' is the intersection of the output set
  // of the else as well as all the 'then' blocks.
  liveValues &= thenLiveOutValues;
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
// DestructorInserter
//===----------------------------------------------------------------------===//

/// Emit a origin end marker for a value that is being consumed.
static void emitLifetimeEnd(Value value, ImplicitLocOpBuilder &builder) {
  // RefLoadOp can only be used on register passable values.  See if this is
  // loading from a var box.
  if (auto load = value.getDefiningOp<RefLoadOp>())
    value = load.getOperand();
  if (auto rebind = value.getDefiningOp<RebindOp>())
    value = rebind.getOperand();

  if (value.getDefiningOp<VarDeclOp>())
    builder.create<VarLifetimeEndOp>(value);
}

static void emitLifetimeEndAfter(Value value, Operation *after) {
  ImplicitLocOpBuilder builder(after->getLoc(), after);
  builder.setInsertionPointAfter(after);
  emitLifetimeEnd(value, builder);
}

namespace {
/// This class holds transient state for the DestructionInsertion pass,
/// accumulating values that need to be destroyed and then emitting and
/// scheduling the destructor calls themselves (potentially mutating the
/// operation with the uses (eg if it is a copyinit).
class DestructorInserter {
public:
  DestructorInserter(ImplicitLocOpBuilder builder, ValueSet &valueSet,
                     std::vector<InFlightDiagnostic> &diagsToEmit)
      : builder(builder), valueSet(valueSet), diagsToEmit(diagsToEmit) {}

  /// This method indicates that the specified value needs to be destroyed after
  /// this operation.  If 'fieldsToDestroy' is non-empty then it specifies which
  /// subfields should be destroyed with zeros, otherwise the whole value needs
  /// to be destroyed.
  void add(Value value, ValueRef valueRef, BitVector fieldsToDestroy = {}) {
    // Look through lit.ref.immut ops to find the underlying mutable thing if
    // we can.  This also helps copy elision which checks for pointer identity.
    value = RefImmutOp::strip(value);
    valuesToDestroy.push_back({value, valueRef, std::move(fieldsToDestroy)});
  }

  enum class DtorEmissionResult {
    /// The destructors were emitted as normal.
    KeepOp,
    /// The operation has been subsumed by a destructor and should be removed.
    RemoveOpWithUse,
  };

  /// This emits any destructors needed at the location specified by the
  /// builder.  If opWithUse is specified, then the inserter is allowed to
  /// perform various optimizations, e.g. if the opWithUse is a copyinit.
  ///
  /// This returns an enum indicating what to do with opWithUse, e.g. if it is
  /// to be deleted by the caller.
  DtorEmissionResult emitDestructors(Operation *opWithUse);

  /// The same as emitDestructors, but there is no opWithUse so no copyinit
  /// elision can happen.
  void emitDestructors() {
    auto result = emitDestructors(/*opWithUse*/ nullptr);
    assert(result == DtorEmissionResult::KeepOp &&
           "should never delete an op if one isn't provided");
    (void)result;
  }

  LLVM_DUMP_METHOD void dump() const;

  /// This is the builder used to insert any destructor calls.
  ImplicitLocOpBuilder builder;

private:
  ValueSet &valueSet;

  /// This is a set of warnings to emit from this pass.  We buffer them and then
  /// emit them at the end of the pass, because dtor insertion is "bottom up"
  /// and we want to emit warnings in a "top down" manner.
  std::vector<InFlightDiagnostic> &diagsToEmit;

  /// During the core op-processing loop, this is the set of values that need to
  /// be destroyed.
  struct ValueToDestroy {
    /// This the SSA value that needs to be destroyed.
    Value value;
    /// The field range covered by value.
    ValueRef valueRef;
    /// If not zero length, this indicates that some subfields are already dead
    /// and the rest need to be destroyed.
    BitVector fieldsToDestroy;
  };
  SmallVector<ValueToDestroy> valuesToDestroy;

  void destroyValueIfNeeded(Value v, ValueRef valueRef,
                            const BitVector &consumedValues,
                            ImplicitLocOpBuilder &builder);
  void emitDestructorCall(Value value, ValueRef valueRef,
                          ImplicitLocOpBuilder &builder);
  DtorEmissionResult optimizeCopyDestroys(Operation *opWithUse);

  enum class CopyInitSuccess {
    Failed,          // Failed to elide.
    Eliminated,      // Eliminated the copyinit entirely.
    ConvertedToMove, // Instruction is still now a moveinit
  };
  CopyInitSuccess elideCopyInitMem(LIT::CallOp copyInitCall, Value copyInitSrc);
  void elideCopyInitReg(LIT::CallOp copyInitCall, Value copyInitSrc);
};
} // end anonymous namespace

void DestructorInserter::dump() const {
  auto &os = llvm::errs();
  os << "Destructor inserter with " << valuesToDestroy.size() << " values\n";
  for (auto &elt : valuesToDestroy)
    os << "  id #" << elt.valueRef.valueId << ": " << elt.value << "\n";
}

/// This emits any destructors needed at the location specified by the
/// builder.  If opWithUse is specified, then the inserter is allowed to
/// perform various optimizations, e.g. if the opWithUse is a copyinit.
///
/// This returns an enum indicating what to do with opWithUse, e.g. if it is
/// to be deleted by the caller.
DestructorInserter::DtorEmissionResult
DestructorInserter::emitDestructors(Operation *opWithUse) {
  // If this is a __copyinit__ call, we can do elision, which may subsume
  // one of our dtors that we need to emit.
  DtorEmissionResult removedOp = optimizeCopyDestroys(opWithUse);

  // TODO: There can be dependencies between dtor calls (e.g. an array of
  // references needs to be destroyed before the elements it references).
  // Sort them before emitting.

  // Emit each value destruction in turn.
  for (auto &v : valuesToDestroy)
    destroyValueIfNeeded(v.value, v.valueRef, v.fieldsToDestroy, builder);

  // Now that we're done, recycle our space for the next iteration.
  valuesToDestroy.clear();
  return removedOp;
}

/// We need to destroy the specified value, which could destroyed as a single
/// destructor call, or could need fieldwise destruction.  Emit the necessary
/// element accesses and calls.
void DestructorInserter::destroyValueIfNeeded(Value value, ValueRef valueRef,
                                              const BitVector &consumedValues,
                                              ImplicitLocOpBuilder &builder) {

  // If we've recursed down to a field that is already fully destroyed, then
  // we're done without further investigation.
  if (!consumedValues.empty() && valueRef.isAllPresent(consumedValues))
    return;

  // If the entire value needs to be destroyed, then emit a destructor for the
  // whole value.  This is the base case for our recursion.
  if (consumedValues.empty() || !consumedValues.test(valueRef.endBit - 1)) {
    // Diagnose an error if a field of the value we must destroy is already
    // destroyed.  We cannot run the destructor on the whole object if one of
    // the fields is missing.
    if (!consumedValues.empty() && !valueRef.isAllMissing(consumedValues)) {
      auto &valueEntry = valueSet.getValueInfo(valueRef.valueId);
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
      diagsToEmit.push_back(std::move(diag));
      return;
    }

    // Ok, the value needs to be dead here.  If we're tracking it and this is
    // a whole object destroy, emit a debug kill.
    if (valueRef.valueId) {
      const ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
      if (info.debugVariable &&
          (consumedValues.empty() ||
           valueRef.getNumBits() == consumedValues.size())) {
        builder.create<DebugInfo::KillOp>(info.debugVariable);
      }
    }

    // Emit the destructor.
    emitDestructorCall(value, valueRef, builder);
    return;
  }

  // Otherwise, we must have an indirect value where some fields are present and
  // some are missing.  Recursively walk the type and destroy just the fields
  // that are missing.
  auto valueType = cast<LIT::StructType>(valueRef.getValueType(value));
  LIT::StructDeclOp structDecl =
      valueSet.typeDeclInfo.getStructDeclForType(valueType);

  // Initialize an evaluator so that we can resolve the field types.
  ParameterEvaluator evaluator;
  for (auto [decl, value] :
       llvm::zip(structDecl.getParams(), valueType.getParamValues()))
    evaluator.setParameterValue(decl, value);

  assert(valueRef.isIndirect && "register values aren't field sensitive");

  unsigned nextBit = 0;
  for (StructFieldOp field : structDecl.getFieldDecls()) {
    auto fieldVal = builder.create<RefStructGEROp>(value, field);
    unsigned numBits = valueSet.typeDeclInfo.getNumFieldsInType(
        evaluator.getReboundType(field.getType()));
    destroyValueIfNeeded(fieldVal, valueRef.getSubfield(nextBit, numBits),
                         consumedValues, builder);

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

/// Given a value of reference type, this checks to see if it is immutable, and
/// casts it back to a mutable reference.  This isn't a generally safe operation
/// from a type system perspective, so should only be used for things like
/// destructor insertion that happen after borrow checking.
static Value getMutableRefForPossiblyImmutValue(Value value,
                                                ImplicitLocOpBuilder &builder) {
  value = RefImmutOp::strip(value);

  // Check to see if the reference is already mutable.
  auto destType = cast<RefType>(value.getType()).getWithMutability(true);
  if (value.getType() == destType)
    return value;

  return builder.create<RebindOp>(destType, value);
}

/// Emit one destructor call for one entire value or field.
///
/// The 'opWithUse' value, if present, is the operation using the overall value
/// being destroyed.  This allows us to perform copy ctor+temp elision.
void DestructorInserter::emitDestructorCall(Value value, ValueRef valueRef,
                                            ImplicitLocOpBuilder &builder) {
  Type destroyedType =
      ValueRef::getDereferencedType(value.getType(), valueRef.isIndirect);
  TypedAttr dtor = valueSet.typeDeclInfo.getDestructorForType(destroyedType);
  if (!dtor) {
    // If there is no destructor, then this is either a trivial type or a
    // non-linear type.  Check for linearTypeErrorMsg and emit it if present.
    if (valueSet.typeDeclInfo.emitErrorMsgIfLinearType(
            builder.getLoc(), destroyedType, diagsToEmit)) {
      valueSet.getValueInfo(valueRef.valueId).hasErrorDiagnosed = true;
    }

    // Otherwise, this is a trivial type; nothing to do.
    return emitLifetimeEnd(value, builder);
  }

  FuncType signature = cast<FuncTypeGeneratorType>(dtor.getType()).getBody();
  assert(signature.getNumResults() == 1 &&
         "dtor should have one result (none type)");
  assert(signature.getNumArguments() == 1 && "dtor should have one operand");

  // We may have a @register_passable value direct (e.g. because it is not in a
  // var).  If so, it needs to be stored into a temporary to invoke the
  // destructor, because it takes it by-ref.
  if (!isa<RefType>(value.getType())) {
    size_t originNum = valueSet.typeDeclInfo.nextAnonOriginNumber++;
    StringAttr originAttr =
        builder.getStringAttr("__dtor_tmp__`" + Twine(originNum));
    auto tmpVar = builder.create<VarDeclOp>(
        value.getType(),
        builder.getStringAttr("__dtor_tmp__" + Twine(originNum)), originAttr,
        VarDeclKind::Implicit);
    builder.create<VarLifetimeStartOp>(tmpVar);
    builder.create<RefStoreOp>(value, tmpVar);
    value = tmpVar;
  }

  // The dtor must take a reference:  Bind the implicit origin of __del__'s self
  // to the origin of the reference we have.
  SmallVector<TypedAttr> implicitOrigins;
  auto delSelfTy = dyn_cast<RefType>(signature.getArguments()[0]);
  if (!delSelfTy) {
    auto diag = mlir::emitError(builder.getLoc())
                << "invalid __del__ that doesn't take register by-ref";
    diagsToEmit.push_back(std::move(diag));
    return;
  }

  value = getMutableRefForPossiblyImmutValue(value, builder);
  auto argRef = cast<RefType>(value.getType());
  assert(delSelfTy.getElementType() == argRef.getElementType());
  implicitOrigins.push_back(argRef.getOrigin());

  // Verify that the address space of the reference matches.  The __del__
  // method will have address space zero.  Attempts to delete other things
  // should not explode the compiler.
  if (delSelfTy.getAddressSpace() != argRef.getAddressSpace()) {
    auto diag = mlir::emitError(builder.getLoc())
                << "cannot destroy value in non-default address space";
    diagsToEmit.push_back(std::move(diag));
    return;
  }

  // Emit the call to the destructor.
  builder.create<LIT::CallOp>(signature.getResults()[0], dtor, implicitOrigins,
                              value);
  emitLifetimeEnd(value, builder);
}

//===----------------------------------------------------------------------===//
// DestructorInserter Copy Elision
//===----------------------------------------------------------------------===//

/// Look to see if the specified operation is a copyinit: if so, check to see
/// if any of the values we're looking to destroy are the input.  If so, try to
/// eliminate the copy in favor of more uses of the now-dead input.
///
///   %tmp = lit.var.decl "anonymous"
///   kgen.call __copyinit__(%src, %tmp)
///   kgen.call __del__(%src)   <<= Thinking about inserting this.
///   kgen.call user(%tmp)      <<= Consuming call.
///
/// If this happens, we want to generate:
///    REMOVED: %tmp = lit.var.decl "anonymous"
///    REMOVED: kgen.call __copyinit__(%src, %tmp)
///    NOTADDED: kgen.call __del__(%src)
///    kgen.call user(%src)      <<= Use %src instead.
///
/// Similar, for a register form, we want to transform:
///    %tmp = kgen.call __copyinit__(%src)
///    kgen.call __del__(%src)   <<= Thinking about inserting this.
///    ...
///    lit.ref.store %tmp, %copy
///    ...
///    kgen.call user(%copy)      <<= Consuming call.
///
/// Into:
///    %tmp = lit.ref.load %src
///    ...
///    lit.ref.store %tmp, %copy
///    ...
///    kgen.call user(%copy)      <<= using call.
DestructorInserter::DtorEmissionResult
DestructorInserter::optimizeCopyDestroys(Operation *opWithUse) {
  auto copyInitCall = dyn_cast_if_present<LIT::CallOp>(opWithUse);
  if (!copyInitCall)
    return DtorEmissionResult::KeepOp;

  // See if we can resolve the callee.
  FnOp callee =
      valueSet.typeDeclInfo.getFuncForSymbol(copyInitCall.getDirectCallee());
  if (!callee ||
      callee.getSpecialFunctionKind() != SpecialFunctionKind::kCopyInit)
    return DtorEmissionResult::KeepOp;

  // Check to see if the copy is immediately destroyed.  If so, we can elide
  // both the copy and the destroy.
  // NOTE: There is a corner case here to be aware of: the copyinit could be
  // the last use of dest (if the result of the copy is dead) the last use of
  // src (what you'd normally think of) as well as the last use of many other
  // values when the input is a reference with an origin set containing
  // multiple things.  We prefer to delete the copy entirely if we can.

  // Handle the register form: `__copyinit__(src) -> T`.  Note that the src is
  // passed in memory.
  Value copySrcMem = RefImmutOp::strip(copyInitCall.getOperand(0));
  if (copyInitCall.getNumOperands() == 1) {
    assert(copyInitCall.getCalleeType().getArgConvention(0) ==
               ArgConvention::ReadMem &&
           "non-trivial register types passed in memory");
    ValueToDestroy *deadSrc = nullptr;
    Value copyDst = copyInitCall.getResult(0);
    for (auto [i, elt] : llvm::enumerate(valuesToDestroy)) {
      if (!elt.fieldsToDestroy.empty())
        continue; // Can only optimize full object destructions.

      // Check to see if the destination is unused.  If so, we can just drop the
      // __copyinit__ entirely.
      if (elt.value == copyDst && !elt.valueRef.isIndirect) {
        Value immSrc = copyInitCall.getOperand(0); // src as immutable reference
        copyInitCall->dropAllReferences();
        valueSet.eraseValueInfo(copyDst);

        // If the input was a lit.ref.immut that is now dead, clean it up.
        if (immSrc.use_empty()) {
          if (auto immut = immSrc.getDefiningOp<RefImmutOp>())
            immut->erase();
        }

        // We're done with this destructor, so remove it from the list.
        valuesToDestroy.erase(valuesToDestroy.begin() + i);
        // Caller will remove the copyinit call.
        return DtorEmissionResult::RemoveOpWithUse;
      }

      // Check to see the copy is the last use of the src value.  If so we can
      // always use the source and avoid a copy.
      if (elt.value == copySrcMem && elt.valueRef.isIndirect)
        deadSrc = &elt;
    }

    // If the source is found to be dead, eliminate it.
    if (deadSrc) {
      elideCopyInitReg(copyInitCall, copySrcMem);

      // We're done with this destructor, so remove it from the list.
      valuesToDestroy.erase(deadSrc);
      // Caller will remove the copyinit call.
      return DtorEmissionResult::RemoveOpWithUse;
    }

    return DtorEmissionResult::KeepOp;
  }

  // Otherwise we have the memory form of `__copyinit__(src, dest)`.
  Value copyDstMem = copyInitCall.getOperand(1);

  // Check to see if the destination is unused.  If so, we can just drop the
  // __copyinit__ entirely.  We need to do this before checking to see if the
  // source is dead.
  ValueToDestroy *deadSrc = nullptr;
  for (auto [i, elt] : llvm::enumerate(valuesToDestroy)) {
    if (!elt.fieldsToDestroy.empty())
      continue; // Can only optimize full object destructions.

    if (elt.value == copyDstMem && elt.valueRef.isIndirect) {
      copyInitCall->dropAllReferences();
      emitLifetimeEndAfter(copyDstMem, copyInitCall);

      // We're done with this destructor, so remove it from the list.
      valuesToDestroy.erase(valuesToDestroy.begin() + i);
      // Caller will remove the copyinit call.
      return DtorEmissionResult::RemoveOpWithUse;
    }

    // Check to see the copy is the last use of the src value, if so, try to
    // use the src directly instead of copying it.
    if (elt.value == copySrcMem && elt.valueRef.isIndirect)
      deadSrc = &elt;
  }

  // If the entire copy isn't dead, but the source is dead, then we can remove
  // it.
  if (deadSrc) {
    DtorEmissionResult result = DtorEmissionResult::KeepOp;
    switch (elideCopyInitMem(copyInitCall, copySrcMem)) {
    case CopyInitSuccess::Failed:
      return DtorEmissionResult::KeepOp;
      // Couldn't elide anything.
    case CopyInitSuccess::Eliminated:
      result = DtorEmissionResult::RemoveOpWithUse;
      break;
    case CopyInitSuccess::ConvertedToMove:
      break; // Remove the dtor, but keep the move.
    }

    // We're done with this destructor, so remove it from the list.
    valuesToDestroy.erase(deadSrc); // valuesToDestroy.begin() + i);
    return result;
  }

  return DtorEmissionResult::KeepOp;
}

/// Return true if the specified 'p1' pointer could point at object or a
/// subcomponent of 'p2'.  This should return true conservatively.
// TODO: In the presence of returned references / origins, we will
// need to be more careful here.
static bool mightPointTo(Value p1, Value p2) {
  assert((isa<PointerType, RefType>(p2.getType())));
  // If the value is an integer or other random thing, then it can't point to
  // anything.
  if (!isa<PointerType, RefType>(p1.getType()))
    return false;

  Value underlyingP1 = OriginTrackable::findUnderlyingValueFromField(p1);
  Value underlyingP2 = OriginTrackable::findUnderlyingValueFromField(p2);
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
//    kgen.call __copyinit__(%src, %tmp)  <<== Last use of %src
// ** kgen.call __del__(%src)   <<== Thinking about inserting this.
//    kgen.call __init__(%src)  <<== Could reinitialize %src before use of %tmp!
//    use(%tmp) use(%src)
//
// Doing this right requires non-trivial liveness analysis which should
// itself be part of a standalone SSA pass post-inlining.  For now we'll
// just catch the most obvious local cases to clean up the IR and provide a
// "guaranteed" optimization.
static bool canEntirelyElideMemoryTemporary(LIT::CallOp copyInitCall,
                                            VarDeclOp tmpDecl) {
  assert(copyInitCall.getOperand(1) == tmpDecl &&
         "the vardecl is known to be directly assigned");
  // Right now we require them to be in the same block, this is overly
  // conservative.
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

      // Ignore lifetime markers.
      if (isa<VarLifetimeStartOp, VarLifetimeEndOp>(user))
        continue;

      if (user->getBlock() != tmpBlock)
        return false; // We don't handle control flow.

      // If we see a lit.ref.immut or rebind of the origin, check all its uses
      // as well.
      if (isa<RefImmutOp, RebindOp>(user)) {
        valuesToCheck.push_back(user->getResult(0));
        continue;
      }

      // Ignore the copyinit of tmp.
      if (user == copyInitCall)
        continue;

      // It may be a lit.load.consume if the value is a register passable type.
      if (auto load = dyn_cast<LoadConsumeOp>(user)) {
        userOfTmp.insert(load);
        continue;
      }

      // Otherwise, the only sort of user we can support is a call.
      auto callUser = dyn_cast<LIT::CallOp>(user);
      if (!callUser)
        return false; // Unknown user.

      // The argument convention for the callee must be consuming or read, not
      // initializing or anything else.  Allowing 'read' allows us to enable
      // this pattern:

      //    %tmp = lit.var.decl "anonymous"
      //    kgen.call __copyinit__(%src, %tmp)  <<== Last use of %src
      // ** kgen.call __del__(%src)   <<== Thinking about inserting this.
      //    use(%tmp)       <= use the temp
      //    consume(%tmp)   <= eventually consume it.
      auto convention = callUser.getCalleeType().getBody().getArgConvention(
          operand.getOperandNumber());
      if (convention != ArgConvention::OwnedMem &&
          convention != ArgConvention::ReadMem)
        return false;
      userOfTmp.insert(callUser);
    }
  }

  // There have to be usersOfTmp: we check to see if the copy is dead before
  // considering this optimization, so the copy itself can't also be dead.
  assert(!userOfTmp.empty() && "tmp should at least be destroyed");

  // Okay, we only see users of the 'tmp' decl that we can understand.  Do a
  // lexical scan to make sure there is nothing between the initialization of
  // the tmp and the use of the tmp that might re-use the source.
  Value srcPointer = copyInitCall.getOperand(0);
  for (auto it = ++Block::iterator(copyInitCall), e = tmpBlock->end();; ++it) {
    // If we ran off the end of the block but we didn't see the users, then the
    // copyinit doesn't dominate this use, something weird is going on, bail
    // out.
    if (it == e)
      return false;

    // We don't recurse into regions, so be conservative.
    if (it->getNumRegions())
      return false;

    // Scan all the operands to see if any of them are related to %src.
    if (llvm::any_of(it->getOperands(),
                     [&](Value v) { return v && mightPointTo(v, srcPointer); }))
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

/// This function handles the case when we see a destructor destroying the src
/// value for a copyinit call.  In these cases, we can just use the source value
/// directly and drop the copy.
void DestructorInserter::elideCopyInitReg(LIT::CallOp copyInitCall,
                                          Value copySrcMem) {
  Value copyDst = copyInitCall.getResult(0);

  // Insert a consuming load after the copyinit (so our dtor walk doesn't
  // see it) that will replace the copy.
  ImplicitLocOpBuilder builder(copyInitCall.getLoc(),
                               &*std::next(Block::iterator(copyInitCall)));
  // TODO: we could get more aggressive and reuse the memory temp when the
  // result is insta-stored if there is some reason to do so.
  auto newResult = builder.create<LoadConsumeOp>(copySrcMem);
  emitLifetimeEndAfter(copySrcMem, newResult);

  copyDst.replaceAllUsesWith(newResult);
  Value immSrc = copyInitCall.getOperand(0); // src as immutable reference
  copyInitCall->dropAllReferences();

  // If the input was a lit.ref.immut that is now dead, clean it up.
  if (immSrc.use_empty()) {
    if (auto immut = immSrc.getDefiningOp<RefImmutOp>())
      immut->erase();
  }

  // The value returned by the copyinit is an owned value, update the
  // ValueSet to know that the LoadConsume is the new value for the
  // ValueID.
  ValueRef ref = valueSet.getDirectValueRef(copyDst, /*isDeref*/ false);
  assert(ref.valueId != 0 && "expected to find the copy value");
  ValueInfo &info = valueSet.getValueInfo(ref.valueId);
  assert(info.value == copyDst);
  info.value = newResult;
}

/// We need to destroy the source for the specified call to a memory-only
/// __copyinit__ call.  Attempt to elide it completely or strength reduce it to
/// a __moveinit__.  The 'copyInitSrc' value is the src operand with
/// lit.ref.immut instructions stripped off.
DestructorInserter::CopyInitSuccess
DestructorInserter::elideCopyInitMem(LIT::CallOp copyInitCall,
                                     Value copyInitSrc) {
  ImplicitLocOpBuilder builder(copyInitCall.getLoc(), copyInitCall);

  // We prefer to completely delete the copy if it is into a temporary location
  // that we can forward.
  //
  // Note: we currently delete explicitly declared temporaries, not just
  // implicit ones.  This is a policy decision, and we should look into
  // the impact on debug information, but generally one wouldn't want debug
  // information to block optimizations.
  if (VarDeclOp tmpDecl =
          copyInitCall.getOperand(1).getDefiningOp<VarDeclOp>()) {
    if (canEntirelyElideMemoryTemporary(copyInitCall, tmpDecl)) {
      // Insert a declaration of the origin for the tmp we're eliding, we know
      // that VarDeclOp's always declare a unique origin.
      auto refType = cast<RefType>(tmpDecl.getType());
      auto param = cast<ParamDeclRefAttr>(refType.getOrigin());

      // The old reference type used a novel origin.  We need to declare it,
      // and coerce back to it with a rebind.
      builder.create<ParamDeclareOp>(ParamDeclAttr::get(param),
                                     AnyOriginAttr::get(param.getType()));
      auto refCasted = builder.create<RebindOp>(tmpDecl.getType(), copyInitSrc);

      // Erase the origin start marker for the temporary. However, keep the
      // origin end markers if the aliased value is a var decl, as they will
      // get "inherited" by the aliased value.
      Value value = OriginTrackable::findUnderlyingValueFromField(refCasted);
      for (Operation *user : llvm::make_early_inc_range(tmpDecl->getUsers())) {
        if (isa<VarLifetimeStartOp>(user)) {
          user->erase();
        } else if (auto end = dyn_cast<VarLifetimeEndOp>(user)) {
          if (value.getDefiningOp<VarDeclOp>())
            end.setOperand(value);
          else
            user->erase();
        }
      }
      tmpDecl.getResult().replaceAllUsesWith(refCasted);

      // We'll delete the copyInit but don't want to invalidate iterators so do
      // later.  Remove the operand uses so we don't see them in later def-use
      // scans, and to make it more obvious when reading IR dumps that these
      // will be gone.
      copyInitCall->dropAllReferences();
      // The caller will remove the copyinit call.
      return CopyInitSuccess::Eliminated;
    }
  }

  auto srcRefType = cast<RefType>(copyInitSrc.getType());
  Type destroyedType = srcRefType.getElementType();

  // Otherwise, try to promote to a __moveinit__ call if present.
  SymbolConstantAttr moveCtor =
      valueSet.typeDeclInfo.getMoveInitForType(destroyedType);
  if (!moveCtor)
    return CopyInitSuccess::Failed;

    // moveCtor must have __moveinit__(out self, owned: Self) type.
#ifndef NDEBUG
  FuncType moveSig = cast<FuncTypeGeneratorType>(moveCtor.getType()).getBody();
  assert(moveSig.getNumArguments() == 2);
  assert(moveSig.getArgConvention(0) == ArgConvention::OwnedMem);
  assert(moveSig.getArgConvention(1) == ArgConvention::ByRefResult);
  auto moveArgs = moveSig.getArguments();
  auto moveValue1Ref = cast<RefType>(moveArgs[0]);
  // srcRefType is immutable here because it was passed to a copy.
  assert(cast<RefType>(moveArgs[1]).getElementType() == destroyedType &&
         moveValue1Ref.getElementType() == destroyedType &&
         moveValue1Ref.isMutableKnown(true));

  auto destType = cast<RefType>(copyInitCall.getOperand(1).getType());
  assert(destType.getElementType() == srcRefType.getElementType());
#endif

  // We know that the input is mutable (otherwise it wouldn't be tracked for
  // destruction), get the reference to a mutable type.
  copyInitSrc = getMutableRefForPossiblyImmutValue(copyInitSrc, builder);
  srcRefType = cast<RefType>(copyInitSrc.getType());

  // Switch the source operand, and update the origin associated with it.
  copyInitCall.setOperand(0, copyInitSrc);
  copyInitCall.setImplicitOrigins(
      {srcRefType.getOrigin(), copyInitCall.getImplicitOrigins()[1]});

  // Transform the copy into a move.
  copyInitCall.setCalleeAttr(moveCtor);
  emitLifetimeEndAfter(copyInitSrc, copyInitCall);
  // We don't want to remove the copyinit, it is now our moveinit.
  return CopyInitSuccess::ConvertedToMove;
}

//===----------------------------------------------------------------------===//
// DestructorInsertion Analysis
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the third pass over a function body, which
/// inserts destructors after the last use of values.
struct DestructorInsertion {
  DestructorInsertion(ValueSet &valueSet) : valueSet(valueSet) {}
  DestructorInsertion(const DestructorInsertion &existing) = delete;
  DestructorInsertion(DestructorInsertion &&existing) = default;

  static DestructorInsertion copy(const DestructorInsertion &existing) {
    DestructorInsertion result(existing.valueSet);
    result.consumedValues = existing.consumedValues;
    result.raiseSet = existing.raiseSet;
    result.breakSet = existing.breakSet;
    result.continueSet = existing.continueSet;
    result.dryRun = existing.dryRun;
    result.functionSignature = existing.functionSignature;
    return result;
  }

  void scanFunction(FnOp func);
  void scanBlock(Block &body);

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op);
  void checkElIfOp(HLCF::ElifOp op);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  BitVector unifyConsumedSets(const BitVector &set1, const BitVector &set2);
  void destroyValuesAtEntryIfNeeded(const BitVector &currentConsumeSet,
                                    Block &block,
                                    const BitVector &fullSetToDestroy,
                                    Location loc);

  void checkConsume(Value value, Operation &op, bool isDeref,
                    DestructorInserter &dtorInserter);
  void checkUse(Value value, bool isDeref, DestructorInserter &dtorInserter);
  void checkDef(Value value, Operation &op, bool isDeref,
                DestructorInserter &dtorInserter);
  void checkOriginEffect(TypedAttr origin, DestructorInserter &dtorInserter);
  bool scheduleNeededDtors(ValueRef use, DestructorInserter &dtorInserter,
                           Value value = Value());

  /// Emit a debug value for the value if it is tracked with debug info.
  void emitDebugInit(Value value, ValueRef valueRef,
                     ImplicitLocOpBuilder &builder);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is the signature of the current function being analyzed.
  FuncTypeGeneratorType functionSignature;

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

  /// This is a set of warnings to emit from this pass.  We buffer them and then
  /// emit them at the end of the pass, because dtor insertion is "bottom up"
  /// and we want to emit warnings in a "top down" manner.
  std::vector<InFlightDiagnostic> diagsToEmit;
};
} // namespace

[[maybe_unused]] void DestructorInsertion::dump() const {
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

void DestructorInsertion::scanFunction(FnOp func) {
  functionSignature = func.getFuncTypeGenerator();

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
  for (auto [argValue, conv] :
       llvm::zip(func.getArguments(),
                 func.getFuncTypeGenerator().getArgConventions())) {
    // Ignore undef-on-input values.
    if (isResultSlot(conv))
      continue;

    bool isIndirect = hasAddress(conv);
    Location loc = argValue.getLoc();
    if (DebugInfo::DISubprogramAttr scope =
            DebugInfo::extractScope(cast<mlir::FunctionOpInterface>(*func)))
      loc = FusedLoc::get(loc.getContext(), {loc}, scope);

    ImplicitLocOpBuilder builder(loc, &funcBody, funcBody.begin());
    DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);
    checkUse(argValue, /*isDeref=*/isIndirect, dtorInserter);
    dtorInserter.emitDestructors();
  }

  // Emit any diagnostics that were queued up in a top-down order.
  while (!diagsToEmit.empty())
    diagsToEmit.pop_back();
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void DestructorInsertion::scanBlock(Block &block) {
  // Process each operation bottom-up in the block.
  SmallVector<std::pair<Value, OperandEffect>> operandEffects;
  SmallVector<ResultEffect> resultEffects;
  SmallVector<TypedAttr> originEffects;

  SmallVector<Operation *> opsToRemove;

  for (Operation &op : llvm::reverse(block)) {
    operandEffects.clear();
    resultEffects.clear();
    originEffects.clear();
    auto overall = getOperationEffects(op, operandEffects, resultEffects,
                                       originEffects, valueSet.originFinder);
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
    case OverallOpValueEffect::elifOp:
      checkElIfOp(cast<HLCF::ElifOp>(op));
      break;
    case OverallOpValueEffect::loopOp:
      checkLoopOp(op);
      break;
    case OverallOpValueEffect::tryOp:
      checkTryOp(cast<LIT::TryOp>(op));
      break;
    }

    // Insert any destructor calls immediately /after/ this operation, since
    // they are for values used by it.
    ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(),
                                 std::next(Block::iterator(&op)));
    DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);

    assert(resultEffects.size() == op.getNumResults() &&
           "getOperationEffects returned wrong # effects");

    for (auto [result, effect] : llvm::zip(op.getResults(), resultEffects)) {
      // CheckUninit pass does all the paranoid checking, don't duplicate it.
      switch (effect) {
      case ResultEffect::ignore:
        continue;
      case ResultEffect::regDefine:
        checkDef(result, op, /*isDeref=*/false, dtorInserter);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true, dtorInserter);
        break;
      case ResultEffect::memDefineUninitToUninit:
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        // If it start/end initialized, emit destructor if already replaced.
        checkUse(result, /*isDeref=*/true, dtorInserter);
        break;
      case ResultEffect::memDefineInitToUninit:
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true, dtorInserter);
        break;
      }
    }

    // Handle all the normal operand and result effects.
    for (auto [operand, effect] : operandEffects) {
      switch (effect) {
      case OperandEffect::regUse:
        checkUse(operand, /*isDeref=*/false, dtorInserter);
        break;
      case OperandEffect::regConsume:
        checkConsume(operand, op, /*isDeref=*/false, dtorInserter);
        break;
      case OperandEffect::memLoad:
        checkUse(operand, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memStoreOwned:
        checkDef(operand, op, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memMut:
        // It is sufficient to just check that we're using the input operation,
        // and if this is the last use of the operation, we should insert a
        // destructor for the value.  checkDef would mark the value as
        // not-live-in, which we don't want.
        checkUse(operand, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memConsume:
        checkConsume(operand, op, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memMarkDestroyed:
        // The lit.ownership.mark_destroyed op consumes the whole object bit of
        // a value only, but not its fields.  This ensures the sub-fields are
        // destroyed but the full object is not.  It is used in destructors
        // primarily.
        if (ValueRef access =
                valueSet.getDirectValueRef(operand, /*isDeref=*/true))
          consumedValues.set(access.endBit - 1);
        break;
      }
    }

    // Process any other indirect origins accessed.
    for (auto origin : originEffects)
      checkOriginEffect(origin, dtorInserter);

    // If the operation used a #lit.any.origin value, then we treat it as an
    // implicit use of all tracked values.  This ensures that the values are not
    // destroyed too early.  Uninit variable scan handles this by adding an
    // attribute with all the value ID's in question.
    if (auto extraUses = op.getAttrOfType<mlir::DenseI32ArrayAttr>(
            extraOriginUsesAttrName)) {
      if (!dryRun)
        op.removeAttr(extraOriginUsesAttrName);

      // Treat this op as using each of the indicated values, putting out a
      // destructor call if this is the last use.
      for (int32_t valueId : extraUses.asArrayRef()) {
        const ValueInfo &info = valueSet.getValueInfo(valueId);
        // NOTE: This can be useful to understand what values are getting
        // lifetime extended and by what.  This is intended more for compiler
        // and library development, not for users.
#if 0
        if (!info.getFullValueRef(valueId).isAllPresent(consumedValues)) {
          auto diag = mlir::emitRemark(op.getLoc());
          if (auto call = dyn_cast<LIT::CallOp>(op))
            diag << "call to '" << call.getDirectCallee() << "'";
          else
            diag << "op";

          diag << " extended with AnyOrigin usage extended lifetime of ";
          if (auto varDecl = info.value.getDefiningOp<VarDeclOp>())
            diag << "'" << varDecl.getName() << "'";
          else
            diag << info.value;
        }
#endif

        checkUse(info.value, /*isDeref*/ info.isIndirect, dtorInserter);
      }
    }

    // Finally emit any enqueued destructors.
    if (dtorInserter.emitDestructors(&op) ==
        DestructorInserter::DtorEmissionResult::RemoveOpWithUse) {
      // If we replaced this operation, remove it after our sweep.
      opsToRemove.push_back(&op);
    }
  }

  // If we had any operations to remove, do so now, simplifying iterator
  // invalidation issues.
  for (Operation *op : opsToRemove)
    op->erase();
}

/// This is returned when the op is a return or unreachable op.
void DestructorInsertion::checkTerminatorOp(Operation &op) {
  consumedValues.reset();
  if (isa<UnreachableOp>(op))
    return;

  assert((isa<KGEN::ReturnOp, ErrorReturnOp>(op)) && "unknown terminator");
  consumedValues.set(0); // Slot 0 indicates that this block is reachable.

  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    // If this value must be live on exit from the function (e.g. a mut
    // argument) demand it.
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

/// 'if' operations propagate the consume sets into each branch, and use the
/// resulting consume sets to make sure the upward propagated set of consumed
/// values is consistent.
void DestructorInsertion::checkIfLikeOp(Operation &ifElseOp) {
  // Given an 'if' like operation (normal 'if' statement or parameter if)
  // perform dtor analysis for each side and insert destructors at the top of
  // the blocks to form a common upward-projected consume set.
  assert(ifElseOp.getNumRegions() == 2 && ifElseOp.getRegion(0).hasOneBlock() &&
         ifElseOp.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");
  BitVector thenConsumedValues = consumedValues;
  scanBlock(ifElseOp.getRegion(0).front());
  // Scan 'else' block.
  thenConsumedValues.swap(consumedValues);
  scanBlock(ifElseOp.getRegion(1).front());

  BitVector merged = unifyConsumedSets(consumedValues, thenConsumedValues);
  if (merged.empty()) // Common case, they are identical.
    return;

  // 'consumedValues' is the current set for the 'else' block, so insert those
  // dtors if needed.
  destroyValuesAtEntryIfNeeded(consumedValues, ifElseOp.getRegion(1).front(),
                               merged, ifElseOp.getLoc());

  // Insert destructors in the 'then' block.
  destroyValuesAtEntryIfNeeded(thenConsumedValues,
                               ifElseOp.getRegion(0).front(), merged,
                               ifElseOp.getLoc());

  // The upward consume set is the union of both sides.
  consumedValues = std::move(merged);
}

// This is used for the HLCF::ElifOp.
void DestructorInsertion::checkElIfOp(HLCF::ElifOp op) {
  // ElIf contains pairs of regions in the elifRegions list, which correspond
  // to a 'condition' and a 'if true' block for each condition.  The live-out
  // set is the intersection of all of the live-out sets for each condition.
  MutableArrayRef<Region> ifRegions = op.getElifRegions();
  assert((ifRegions.size() % 2) == 0 && "Must have pairs of regions");

  // Destructor insertion is a backward pass, so we process the else to see the
  // consumed set coming in, then process each if/then pair as merging with its
  // consume set.
  BitVector thenExitConsumedValues = consumedValues;
  Block *elseBlock = &op.getElseRegion().front();
  scanBlock(*elseBlock);

  // For each `if cond: then else: ..` block, we have a consumed value set for
  // the else, which we have to unify with this then block before we can
  // continue up the if/else chain.
  for (size_t i = ifRegions.size(); i != 0; i -= 2) {
    Block &condBlock = ifRegions[i - 2].front();
    Block &thenBlock = ifRegions[i - 1].front();

    // Process the 'then' block with the consume set from after the 'if' chain.
    BitVector elseConsumeSet = std::move(consumedValues);
    consumedValues = thenExitConsumedValues;
    scanBlock(thenBlock);

    // We now have the consume set from the 'then' and else'.  Merge these
    // two sets, and if they differ, insert destructor calls.
    BitVector merged = unifyConsumedSets(consumedValues, elseConsumeSet);
    if (!merged.empty()) { // In the common case, they are identical.
      // 'consumedValues' is the current set for the 'then' block, so insert
      // those dtors if needed.
      destroyValuesAtEntryIfNeeded(consumedValues, thenBlock, merged,
                                   op.getLoc());

      // Insert destructors in the 'else' block.
      destroyValuesAtEntryIfNeeded(elseConsumeSet, *elseBlock, merged,
                                   op.getLoc());

      // The upward consume set is the union of both sides.
      consumedValues = std::move(merged);
    }

    // After the 'then' and 'else' blocks are unified, we need to scan the
    // 'cond' block to see which one was picked.  The condition block contains
    // an arbitrary expression which can be the last use of various values, so
    // it gets destructors inserted as well.
    scanBlock(condBlock);

    // For the next 'if cond: then' block, this condition is the effective else
    // block.
    elseBlock = &condBlock;
  }

  // At the end, the upwardly demanded set for the whole statement is what the
  // statement demands.
}

/// Given two consume sets that correspond to an 'if-like' construct which
/// diverges control flow, compute the union of the two consume sets and return
/// it, or RETURN AN EMPTY BITVECTOR if they are identical.
///
/// Consider:     if cond: use(a) else: use(b)
///
/// In this case, the 'then' block will use "a" and the else block will use "b".
/// This returns the union of both {a,b}.  This union operation is non-trivial
/// in other cases though.
///
BitVector DestructorInsertion::unifyConsumedSets(const BitVector &set1,
                                                 const BitVector &set2) {
  // If they agree already, then there is nothing to do.
  if (set1 == set2)
    return BitVector();

  // We don't want to perform meets with unreachable code (e.g. from `if False:
  // stuff`: if either of the regions is unreachable, then propagate the other
  // one.  This matters because there is no conservative "missing" set for whole
  // object bits.  We use the sentinel's consume bit to know if anything is
  // consumed.
  if (!set1[0]) // If "then" isn't reachable, return "else".
    return set2;
  if (!set2[0]) // If "else" isn't reachable, return "then".
    return set1;

  // Given two consume sets, our upward propagated final set will be the
  // union of both sets.
  BitVector result = set1;
  result |= set2;

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
  // If we see this, have the union set demand the whole object so it can be
  // destroyed.
  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    // If the whole-object consume bits agree on both sides, then there is
    // nothing to do.
    if (!valueInfo.isIndirect)
      continue; // Register values have a single bit.

    // If the whole object is already destroyed on both sides, then we don't
    // have to worry about this.  It may be consuming subobjects at a time.
    auto endBit = valueInfo.endValueBit - 1;
    if (set1[endBit] && set2[endBit])
      continue;

    // If any subfields are consumed, then we consume the whole object so the
    // destructor can be run.
    ValueRef ref(/*index*/ 0, valueInfo.startValueBit, valueInfo.endValueBit,
                 valueInfo.isIndirect);
    // If no part of this value is consumed, then ignore it.
    if (ref.isAllMissing(result))
      continue;

    // If this is a merge between 'self' which is not consumed at all on one
    // side, and is consumed a bit on the other side, ignore this and propagate
    // up the simple union.  This happens in error handling scenarios because
    // the error result doesn't demand anything (not even the full object bit)
    // but the other path can demand a partially initialized set of stuff.
    if (valueInfo.isFullObjectLiveOnEntry)
      continue;

    // Otherwise, some part is required, so require the whole thing on both
    // sides so it can be destroyed.
    result.set(valueInfo.startValueBit, valueInfo.endValueBit);
  }

  return result;
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

  // The original set will be what any 'break' statement sees.
  loopBodySets.breakSet = &breakSet;

  // If there is an 'else' on a @parameter for, process it to determine the
  // consume set going into the bottom of the loop.
  BitVector elseBlockConsumeSet;
  if (loopOp.getNumRegions() == 2 && isParamFor) {
    scanBlock(loopOp.getRegion(1).front());
    // Save the set of values consumed by the 'else' block for later.  It is
    // possible that the loop will consume more values and we'll need to insert
    // destructor calls into the else.
    elseBlockConsumeSet = consumedValues;
  }

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
    auto merged = unifyConsumedSets(continueSet, loopBodySets.consumedValues);
    if (!merged.empty())
      loopBodySets.consumedValues = std::move(merged);
    continueSet = loopBodySets.consumedValues;

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
    loopBodySets.scanBlock(loopOp.getRegion(0).front());

    // If we are a '@parameter for' with an else block, the loop body may have
    // more demands than the else block does.  Make sure we destroy these values
    // in the else block if needed.
    if (!elseBlockConsumeSet.empty()) {
      destroyValuesAtEntryIfNeeded(
          elseBlockConsumeSet, loopOp.getRegion(1).front(),
          loopBodySets.consumedValues, loopOp.getLoc());
    }
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
void DestructorInsertion::checkConsume(Value value, Operation &op, bool isDeref,
                                       DestructorInserter &dtorInserter) {
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
    ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
    if (info.hasErrorDiagnosed)
      return;
    ValueRef fullValueRef = info.getFullValueRef(valueRef.valueId);

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
    diagsToEmit.push_back(std::move(diag));
    info.hasErrorDiagnosed = true;
  }

  valueRef.markBits(consumedValues, true);

  if (!dryRun) {
    ImplicitLocOpBuilder builder(op.getLoc(), &op);

    /// Emit a debug kill marker for the value if it is tracked with debug info
    /// an if full value is destroyed.
    // TODO(#34115): Emit fragment end-of-life for partial destruction.
    const ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
    if (info.debugVariable && valueRef.startBit == info.startValueBit &&
        valueRef.endBit == info.endValueBit) {
      builder.create<DebugInfo::KillOp>(info.debugVariable);
    }

    emitLifetimeEndAfter(value, &op);
  }
}

/// Check a use of a value.  Iff this is the /last/ use of the value, emit a
/// destructor of the overall value.  The 'opWithUse' value (if present)
/// indicates the operation performing the use.  This enables copy ctor elision,
/// but this is null at the start of block/function for example.
void DestructorInsertion::checkUse(Value value, bool isDeref,
                                   DestructorInserter &dtorInserter) {
  // If this is a direct reference to a value, we are tracking it, meaning
  // there are dedicated bits in the consumedValues bitvector that represent
  // the consumption state of this value.
  if (ValueRef access = valueSet.getDirectValueRef(value, isDeref)) {
    (void)scheduleNeededDtors(access, dtorInserter, value);
    return;
  }

  // We are not tracking this value directly, it could be tied to an origin
  // declared by a value we do track. If this is the case, check these values
  // for destruction.
  for (ValueRef access : valueSet.getValueRefsForAccess(value, isDeref)) {
    // Do not pass "value" here, because it will refer to the reference, which
    // may not be to the actual tracked value for 'access'.  For example, in
    // 'use(cond ? a : b)' we want to think about "a" and "b".
    (void)scheduleNeededDtors(access, dtorInserter);
  }
}

/// This operation defines the specified value.  If the value is dead on
/// arrival, emit a destructor of the value.
void DestructorInsertion::checkDef(Value value, Operation &op, bool isDeref,
                                   DestructorInserter &dtorInserter) {
  // If there is no use of the value we are defining, scheduleNeededDtors will
  // emit a dtor after the op. This happens when we have things like:
  //
  //   init(&aggregate)
  //   ...
  //   aggregate.field1 = newValue  <<-- we are here
  if (ValueRef direct = valueSet.getDirectValueRef(value, isDeref)) {
    bool isFullUseDestroy = scheduleNeededDtors(direct, dtorInserter, value);

    if (!dryRun && value.getDefiningOp<VarDeclOp>()) {
      // Emit this above the operation.
      ImplicitLocOpBuilder builder(op.getLoc(), &op);
      emitDebugInit(value, direct, builder);
      builder.create<VarLifetimeStartOp>(value);
    }

    // If the destroyed value is a user-defined value that was just defined,
    // warn about the useless store.
    if (!dryRun && isFullUseDestroy) {
      ValueInfo &valueEntry = valueSet.getValueInfo(direct.valueId);
      // Don't warn about assignments into synthesized temporaries or arguments.
      auto varDecl = valueEntry.value.getDefiningOp<VarDeclOp>();
      if (varDecl && varDecl.shouldWarnAboutUnused()) {
        auto diag = mlir::emitWarning(op.getLoc()) << "assignment to ";
        BitVector allMissing(consumedValues.size(), true);
        direct.markBits(allMissing, false);
        addBadValueNameToDiag(direct, allMissing, valueSet, diag);
        diag << " was never used; assign to '_' instead?";
        diagsToEmit.push_back(std::move(diag));
      }
    }

    direct.markBits(consumedValues, false);
    return;
  }

  // For indirect references, we treat this as a use, which will insert dtor
  // calls if this was the last use of any indirectly referenced values.
  checkUse(value, isDeref, dtorInserter);

  // Otherwise, we need to direct-emit a destructor call of the reference
  // itself since this operation will overwrite the value and we can't model
  // it in a field sensitive way.  The uninitialized checker verified that the
  // value is guaranteed live-in when nontrivial and indirect.
  if (!valueSet.isTrivial(value, isDeref) && !dryRun) {
    // Destructor call goes ahead of the mutation, not after.
    ImplicitLocOpBuilder builder(op.getLoc(), &op);
    DestructorInserter beforeDtorInserter(builder, valueSet, diagsToEmit);
    beforeDtorInserter.add(value, /*Just do it*/ ValueRef(0, 0, 0, isDeref));
    beforeDtorInserter.emitDestructors();
  }
}

/// Check any unstructured origins that are accessed by the operation.
void DestructorInsertion::checkOriginEffect(TypedAttr origin,
                                            DestructorInserter &dtorInserter) {
  // Iff this is the /last/ use of the value, emit a dtor for the value.
  for (auto access : valueSet.getValueRefsForOrigin(origin))
    (void)scheduleNeededDtors(access, dtorInserter);
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

  auto valueDRType = dyn_cast<LIT::StructType>(valueType);
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

/// Given a use of a value or subfield, figure out the maximal unconsumed
/// subfield that contains it.  For example, in:
///
///   init(&aggregate)
///   use(&aggregate.field1.subfield)  <<-- We are here.
///   # Should insert: __del__(aggregate.field1)
///   init(&aggregate.field1)
///   __del__(&aggregate)
///
/// we want to return "aggregate.field1", not subfield.
static std::pair<SmallVector<StructFieldOp>, ValueRef>
computeAccessPathForMaxUnconsumedField(ValueRef use,
                                       const BitVector &consumedValues,
                                       const ValueInfo &valueInfo,
                                       TypeDeclInfo &typeDeclInfo) {
  // This only applies to indirect uses.
  if (!use.isIndirect)
    return {{}, use};

  Type valueType = cast<RefType>(valueInfo.value.getType()).getElementType();

  // Figure out where the use is WITHIN the value.
  ValueRef fullValueRef = valueInfo.getFullValueRef(use.valueId);
  unsigned numValueBits = fullValueRef.getNumBits();
  ValueRef useWithinValue = use.getWithoutBaseOffset(fullValueRef.startBit);
  unsigned totalOffset = fullValueRef.startBit;

  // Drill down into this value until we find something that isn't consumed.
  SmallVector<StructFieldOp> result;
  while (consumedValues[totalOffset + numValueBits - 1]) {
    // Okay, we must be accessing some subfield of this total value.  Figure out
    // which one, it must be field sensitive.
    auto [fieldDecl, fieldStartBit, fieldNumBits] =
        typeDeclInfo.getFieldContaining(cast<LIT::StructType>(valueType),
                                        useWithinValue.startBit);

    // Don't drill into the subfield if we're spanning multiple of them.
    if (useWithinValue.getNumBits() > fieldNumBits)
      break;

    // We're drilling into this field.
    result.push_back(fieldDecl);
    useWithinValue = useWithinValue.getWithoutBaseOffset(fieldStartBit);
    totalOffset += fieldStartBit;
    numValueBits = fieldNumBits;
    valueType = fieldDecl.getType();
  }

  return {std::move(result),
          ValueRef(use.valueId, totalOffset, totalOffset + numValueBits,
                   /*isIndirect=*/true)};
}

/// There is a use of the specified 'use' portion of a live value.  If this
/// is the last use of some value, schedule a destructor to clean it up.
/// 'value' is an optional value indicating the MLIR value corresponding to
/// this, which is useful to avoid emitting redundant lit.struct.ger
/// instructions when we already have it.
///
/// Returns true if the destructor was scheduled to destroy the entire use.
bool DestructorInsertion::scheduleNeededDtors(ValueRef use,
                                              DestructorInserter &dtorInserter,
                                              Value value) {
  assert(use && "Only works on valid refs");

  // If the accessed value had an error already or nothing in this value needs
  // destroying, then ignore the request.
  ValueInfo &valueInfo = valueSet.getValueInfo(use.valueId);
  if (valueInfo.hasErrorDiagnosed || use.isAllPresent(consumedValues))
    return false;

  // If we are just computing the consumedValue set, don't actually insert any
  // destructor calls.
  if (dryRun) {
    use.markBits(consumedValues, true);
    return false;
  }

  // 'self' in an initializer is modeled as having its whole-object bit set
  // on entry to the function, but the fields may be in partially initialized
  // states throughout the body of the initializer.  We only treat the full
  // object as being initialized if all of its fields are.  This allows the
  // definition and rewrite of 'self' to work correctly, but doesn't try to
  // run the destructor on a partially initialized self.
  if (valueInfo.isFullObjectLiveOnEntry &&
      valueInfo.endValueBit == use.endBit &&
      valueInfo.startValueBit == use.startBit) {
    // If some of the fields are already missing, don't destroy self.
    --use.endBit;
    if (!use.isAllMissing(consumedValues))
      consumedValues[use.endBit] = true;
    ++use.endBit;

    // If this was the only missing bit, then we're good.
    if (use.isAllPresent(consumedValues))
      return false;
  }

  // Check to see if this whole value needs to be destroyed.
  bool isFullObjectDestroy = !consumedValues[use.endBit - 1];

  // If this is the last use of some subfield of a value that needs to be
  // destroyed, emit a destructor for the WHOLE overall value.
  //
  //   init(&aggregate)
  //   use(&aggregate.field1)
  //   use(&aggregate.field2.subelt)  <<-- We are here.
  //   # Should insert: __del__(&aggregate)
  //
  // In this case, we destroy the overall value.  However, we may be in a field
  // sensitive case where the subfield is getting reinitialized, e.g.:
  //
  //   init(&aggregate)
  //   use(&aggregate.field1.subfield)  <<-- We are here.
  //   # Should insert: __del__(aggregate.field1)
  //   init(&aggregate.field1)
  //   __del__(&aggregate)
  //
  // we have to destroy 'aggregate.field1'.  Figure out what access path we need
  // to destroy.
  auto [accessPath, adjustedUse] = computeAccessPathForMaxUnconsumedField(
      use, consumedValues, valueInfo, valueSet.typeDeclInfo);

  // If we were passed in a field that matches what we need, use it to avoid
  // inserting additional GER operations.  Otherwise we re-derive from the root.
  if (!value || use != adjustedUse) {
    value = valueInfo.value;
    use = adjustedUse;

    // Drill into the right field.
    for (StructFieldOp subfield : accessPath)
      value = dtorInserter.builder.create<RefStructGEROp>(value, subfield);
  }

  // Get the type for the value so we can poke at it.
  // If a generic type or trivial, then emit a destructor call (or nothing).
  auto valueType = dyn_cast<LIT::StructType>(use.getValueType(value));
  if (!valueType) {
    // We are going to emit a destructor for the specified ValueRef, so all none
    // of the things we are about to destroy should already be destroyed.
    assert(use.isAllMissing(consumedValues) &&
           "cannot have partially consumed object");
    dtorInserter.add(value, use);
    use.markBits(consumedValues, true);
    return isFullObjectDestroy; // Destroyed the full value.
  }

  // Trivial types don't have __del__ methods and can't be tracked, so if
  // this is referring to one of them, make sure to clear the bits so we
  // don't think they need to be destroyed.
  clearTrivialFields(use, valueType, consumedValues, valueSet);

  // If we need to destroy the whole value, we can just use an empty BitVector,
  // otherwise we need to specify which subelements are to be destroyed, so we
  // copy it.
  BitVector fieldsToDestroy;
  if (!use.isAllMissing(consumedValues))
    fieldsToDestroy = consumedValues;
  dtorInserter.add(value, use, std::move(fieldsToDestroy));
  use.markBits(consumedValues, true);

  // Return true if we destroyed the full reference.
  return isFullObjectDestroy;
}

void DestructorInsertion::emitDebugInit(Value value, ValueRef valueRef,
                                        ImplicitLocOpBuilder &builder) {
  assert(!dryRun && "shouldn't be called in a dry run");
  ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
  // Insert debug value if full value is initialized.
  if (info.debugVariable && valueRef.startBit == info.startValueBit &&
      valueRef.endBit == info.endValueBit) {
    // The IR type needs to be deref'ed to get the source type. Encode the IR
    // type as a pointer type.
    auto newIrValue = DebugInfo::DIIRValueExprAttr::get(value.getType());
    auto conversion = DebugInfo::DIDerefExprAttr::get(
        newIrValue,
        cast<DebugInfo::DIUnresolvedMLIRType>(info.debugVariable.getType())
            .getType());
    builder.create<DebugInfo::ValueOp>(value, info.debugVariable, conversion);
  }
}

/// Insert destructors calls into the start of 'block' for objects in the
/// 'fullSetToDestroy' that are not already in the 'currentConsumeSet'.  This is
/// used at control flow merges.
///
/// This does not modify 'consumedValues', and does respect 'dryRun'.
void DestructorInsertion::destroyValuesAtEntryIfNeeded(
    const BitVector &currentConsumeSet, Block &block,
    const BitVector &fullSetToDestroy, Location loc) {
  // If we are in a dry run or the two sets match, or the block is unreachable,
  // don't actually insert anything.
  if (dryRun || currentConsumeSet == fullSetToDestroy ||
      isa<UnreachableOp>(block.front()))
    return;

  // entriesToDestroy = fullSetToDestroy & ~currentConsumeSet.
  BitVector entriesToDestroy = fullSetToDestroy;
  entriesToDestroy.reset(currentConsumeSet);

  // Move consumedValues out of the way so we don't break it.  We need to use
  // scheduleNeededDtors below, which is hard coded to mutate consumedValues.
  BitVector savedConsumedValues = std::move(consumedValues);

  // We *only* want to destroy the values in entries, not any other values that
  // may be partially overlapped, so mark all the other things as "already
  // destroyed".  This is to work with 'scheduleNeededDtors'.
  assert(&entriesToDestroy != &consumedValues &&
         "This logic doesn't work when passed 'consumedValues' directly");
  consumedValues = entriesToDestroy;
  consumedValues.flip();

  // As we scan through bits, we walk through corresponding ValueInfos to know
  // what we are working with.
  MutableArrayRef<ValueInfo> valueInfos = valueSet.getValueInfos();
  size_t nextValueInfo = 0;

  // Any dtor calls will be emitted at the start of the block.
  DestructorInserter dtorInserter(
      ImplicitLocOpBuilder(loc, &block, block.begin()), valueSet, diagsToEmit);

  int nextToDestroy = entriesToDestroy.find_first();
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
    (void)scheduleNeededDtors(fullValueRef, dtorInserter);

    // Find the next object to destroy.
    nextToDestroy = entriesToDestroy.find_next(fullValueRef.endBit - 1);
  }

  dtorInserter.emitDestructors();

  // Restore consumedValues.
  consumedValues = std::move(savedConsumedValues);
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
    CachedOriginFinder originFinder;

    // TODO: Do in parallel, watch out for mutations of TypeDeclInfo and
    // originFinder though!
    bool hadError = false;
    for (auto func : functionVector)
      hadError |= failed(processFunction(func, typeDeclInfo, originFinder));

    if (hadError)
      return signalPassFailure();
  }

  LogicalResult processFunction(FnOp func, TypeDeclInfo &typeDeclInfo,
                                CachedOriginFinder &originFinder);
};
} // namespace

LogicalResult
CheckLifetimes::processFunction(FnOp func, TypeDeclInfo &typeDeclInfo,
                                CachedOriginFinder &originFinder) {

  // If the function is a trait function or something else unreachable, we don't
  // need to process it.
  Block &funcBody = func.getFunctionBody().front();
  if (isa<UnreachableOp>(funcBody.front()))
    return success();

  // Walk #1: Collect all of the values declared in the function that have
  // ownership to track, and number them.
  ValueSet valueSet(typeDeclInfo, func, originFinder);

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined, emitting diagnostics as we go.
  UninitializedValueScan(valueSet).scanFunction(func);

  // Walk #3: Scan the function bottom-up, inserting destructor calls, inserting
  // lifetime markers, and eliding temporaries.
  DestructorInsertion(valueSet).scanFunction(func);

  // Now that we've transformed the function, look for any vardecls that only
  // have lifetime markers.  They can be removed, because all their uses got
  // forwarded or rewritten.
  for (ValueInfo &info : valueSet.getValueInfos()) {
    if (!info.value) // Already removed value.
      continue;

    auto varDecl = info.value.getDefiningOp<VarDeclOp>();
    if (!varDecl)
      continue;

    if (!info.isEverUsed && varDecl.shouldWarnAboutUnused()) {
      mlir::emitWarning(varDecl.getLoc())
          << "variable '" << varDecl.getName().str()
          << "' was never used, remove it?";
    }

    // Check to see if there are any uses other than lifetime markers.
    bool hasInterestingUse = false;
    for (Operation *user : varDecl->getUsers()) {
      if (isa<VarLifetimeStartOp, VarLifetimeEndOp, RebindOp, RefImmutOp>(
              user) &&
          user->use_empty())
        continue;

      hasInterestingUse = true;
      break;
    }
    if (hasInterestingUse)
      continue;

    // Okay, nothing interesting happening here.  Remove any lifetime markers
    // and remove the vardecl as well.
    while (!varDecl->use_empty())
      varDecl->user_begin()->erase();
    varDecl->erase();
  }

  // Return failure if we generated errors for any of the tracked values.
  return failure(llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  }));
}
