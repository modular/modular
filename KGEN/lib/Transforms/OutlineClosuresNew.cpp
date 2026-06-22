//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "outline-closures-new"

namespace M::KGEN {
#define GEN_PASS_DEF_OUTLINECLOSURESNEW
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

struct OutlineClosuresNewPass
    : M::KGEN::impl::OutlineClosuresNewBase<OutlineClosuresNewPass> {
  using OutlineClosuresNewBase::OutlineClosuresNewBase;

  void runOnOperation() override;
};

namespace {
struct Capture {
  /// the symbol of the copy/move constructor, if applicable
  std::optional<TypedAttr> moveOrCopySym;
  /// the symbol of the destructor, if applicable
  std::optional<TypedAttr> delSym;

  /// the source of the capture. This is the value used to create a copy if the
  /// symbol is nonnull.
  Value origin;
};
} // namespace

static Value allocateHeapMemory(PointerType ptrType, OpBuilder &b,
                                Location loc) {
  TypedAttr elementType = TypeParamAttr::get(
      ptrType.getElementType(), TypeType::get(ptrType.getContext()));
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value sizeOf = ParamConstantOp::create(
      b, loc, ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = ParamConstantOp::create(
      b, loc, ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return POP::AlignedAllocOp::create(b, loc, ptrType,
                                     ValueRange{alignOf, sizeOf});
}

static std::optional<std::pair<TypeGeneratorRefAttr, StructInstanceType>>
getTypeValuePathData(ClosureInitOp closureInit,
                     StructGeneratorOp structGeneratorOp) {
  auto typeValue = closureInit->getAttrOfType<TypedAttr>("typeValue");
  if (!typeValue)
    return std::nullopt;
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return std::nullopt;
  auto typeValueType = dyn_cast<TypeValueType>(typeParam.getTypeValue());
  if (!typeValueType)
    return std::nullopt;
  auto typeGeneratorRef =
      dyn_cast<TypeGeneratorRefAttr>(typeValueType.getTypeValue());
  if (!typeGeneratorRef)
    return std::nullopt;
  auto structInstanceType =
      dyn_cast<StructInstanceType>(structGeneratorOp.getValueDomainType());
  if (!structInstanceType)
    return std::nullopt;
  return std::make_pair(typeGeneratorRef, structInstanceType);
}

/// Given a location, erase any parameter references by inserting unresolved
/// type for every input type. TODO: recreate a resolved version. This change
/// unblocks using synthesized code from parametric blocks in debug builds
/// because the synthesized code does not depend on the parameters but the
/// parameters are embedded in the locations, resulting in dangling parameter
/// references.
static Location stripParameterRefsFromLoc(Location loc) {
  auto fusedLoc = dyn_cast<FusedLoc>(loc);
  if (!fusedLoc)
    return loc;

  auto subprogram =
      dyn_cast_if_present<DebugInfo::DISubprogramAttr>(fusedLoc.getMetadata());
  if (!subprogram)
    return loc;

  // Create an empty function type (no inputs, no outputs)
  auto strippedType = DebugInfo::DISubroutineType::get(
      loc.getContext(), SmallVector<DebugInfo::DIType>(),
      SmallVector<DebugInfo::DIType>());
  if (!strippedType)
    return loc;

  auto newSubprogram = DebugInfo::DISubprogramAttr::get(
      subprogram.getCompileUnit(), subprogram.getScope(),
      subprogram.getSourceName(),
      StringAttr::get(
          subprogram.getContext(),
          llvm::Twine(subprogram.getLinkageName().getValue(), "_auxiliary")),
      subprogram.getFile(), subprogram.getLine(), subprogram.getScopeLine(),
      subprogram.getSubprogramFlags(),
      cast<DebugInfo::DISubroutineType>(strippedType));

  return FusedLoc::get(fusedLoc.getContext(), fusedLoc.getLocations(),
                       newSubprogram);
}

namespace {
/// The ClosureLifter is responsible for
/// (a) lifting a closure init into a top level function + capture struct and
/// (b) storing metadata necessary to replace references to the closure.
struct ClosureLifter {
  ClosureLifter(SymbolTable &symtab, bool debugBuild)
      : counter(0), symtab(symtab), debugBuild(debugBuild) {}
  /// Given a closure init op, generate functions for call, copy, move, and del
  /// + struct instance to store captures.
  LogicalResult liftClosureInit(ClosureInitOp closureInit,
                                GeneratorOp generator,
                                StructGeneratorOp generatorOp);

  /// Symbol name uniquer requires a counter.
  unsigned counter;

  /// The symbol table of the module.
  SymbolTable &symtab;

  /// Pair a parameter value with the closure attr so that we can replace the
  /// abstraction with the calculated type. For example, we may calculate that
  /// the closure "fn" inside parent @foo captured three parameters A,B,C in
  /// which case we'd like to replace the attribute #kgen.closure<@foo "fn">
  /// with the struct attr <{A,B,C}>.
  struct ClosureParentKey {
    SymbolRefAttr parent;
    StringAttr nestedName;
  };
  DenseMap<ClosureParentKey, llvm::SetVector<ParamDeclAttr>>
      paramCaptureToStructAttr;

  /// Pair a (parent symbol, nested function name, method) with the symbol of
  /// the lifted function so that closure symbols can be replaced robustly even
  /// if types embedded in attributes are rewritten later.
  struct ClosureMethodKey {
    SymbolRefAttr parent;
    StringAttr nestedName;
    ClosureMethodAttr method;
  };
  DenseMap<ClosureMethodKey, SymbolConstantAttr> liftedClosureSymbols;
  /// Pair the closure type with the struct type of the generated capture struct
  /// so that the closure types can be replaced.
  DenseMap<ClosureType, Type> closureTypeToStructTypes;
  DenseMap<ClosureType, Type> closureTypeToStructInstTypes;
  DenseMap<ClosureType, StructGeneratorOp> closureTypeToStructGen;

  /// True if built with debug metadata.
  bool debugBuild;
  struct ClosureInitData {
    ClosureInitData(llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
                    ClosureType closureType, ClosureInitOp closureInit,
                    StructGeneratorOp structGeneratorOp, GeneratorOp generator,
                    SmallVector<TypedAttr> &&moveSymbols,
                    std::optional<SmallVector<TypedAttr>> &&copySymbols);
    Type selfType(Type loweredClosureType) const {
      return closureType.getClosureMemoryKind() == ClosureMemoryKind::TRIVIAL
                 ? loweredClosureType
                 : PointerType::get(loweredClosureType);
    }
    StringRef regionName() const { return closureType.getName(); }
    GeneratorOp getGenerator() const { return generator; }
    ClosureType getClosureType() const { return closureType; }
    ClosureInitOp getClosureInit() const { return closureInit; }
    bool isEscaping() const {
      return closureType.getClosureMemoryKind() == ClosureMemoryKind::ESCAPING;
    }
    bool isMem() const {
      return closureType.getClosureMemoryKind() != ClosureMemoryKind::TRIVIAL;
    }
    bool isFlattenedSingletonClosure(size_t captureCount) const {
      return captureCount == 1 &&
             closureType.getClosureMemoryKind() !=
                 ClosureMemoryKind::ESCAPING &&
             closureType.getClosureMemoryKind() !=
                 ClosureMemoryKind::NONESCAPING;
    }
    ArrayRef<ParamDeclAttr> getCapturedParamDecls() const {
      return ArrayRef(capturedParamDecls.begin(), capturedParamDecls.end());
    }
    ArrayRef<TypedAttr> getMoveSymbols() const {
      return ArrayRef(moveSymbols.begin(), moveSymbols.end());
    }
    ArrayRef<TypedAttr> getCopySymbols() const {
      return ArrayRef(copySymbolsMaybe->begin(), copySymbolsMaybe->end());
    }
    bool isCopyable() const { return copySymbolsMaybe.has_value(); }
    ClosureSymbolAttr closureSymbolForSourceName(ClosureMethod method) const;
    Location getLiftedLocation() const { return liftedLocation; }

  private:
    llvm::SetVector<ParamDeclAttr> capturedParamDecls;
    /// The map of symbols to replace.
    DenseMap<ClosureMethod, ClosureSymbolAttr> abstractSymbolMap;
    ClosureType closureType;
    ClosureInitOp closureInit;
    GeneratorOp generator;
    SmallVector<TypedAttr> moveSymbols;
    std::optional<SmallVector<TypedAttr>> copySymbolsMaybe;
    ParamClosureType paramClosureType;
    Location liftedLocation;
  };

private:
  /// Lift a register passable closure. The characterization is in the lifted
  /// signature: a register passable closure's lifted call function has an
  /// implicit self argument of struct type.
  Value liftRegPassableClosure(OpBuilder &b, ClosureInitData &data,
                               ArrayRef<Capture> captureMechanisms,
                               Type loweredClosureType,
                               Type loweredClosureInstType);
  /// Lift a closure with no captures. We can skip the loading/storing of
  /// captures and use the lowered closure type for the synthesized self.
  Value liftThinClosure(OpBuilder &b, ClosureInitData &data,
                        bool isRegisterPassable, Type loweredClosureType,
                        Type loweredClosureInstType);
  /// Lift a non-register passable closure. The characterization is in the
  /// lifted signature: a non-register passable closure's lifted call function
  /// has an implicit self argument of pointer type.
  Value liftNonRegPassableClosure(OpBuilder &b,
                                  ClosureInitData &closureInitData,
                                  ArrayRef<Capture> captureMechanisms,
                                  Type loweredClosureType,
                                  Type loweredClosureInstType);
  void liftMoveOrCopyFunction(OpBuilder &b, Location loc, ClosureInitData &data,
                              Type loweredClosureType,
                              ArrayRef<Capture> captureMechanisms, bool isMove);
  void liftDelFunction(OpBuilder &b, Location loc, ClosureInitData &data,
                       Type loweredClosureType,
                       ArrayRef<Capture> captureMechanisms);

  void
  createClosureGenerator(OpBuilder &b, Location location,
                         ClosureInitData &closureInitData, ClosureMethod method,
                         FunctionType funcType,
                         llvm::function_ref<void(GeneratorOp)> populateBody,
                         ArrayRef<ArgConvention> argConventions);

  /// Given closure metadata the captures, emit code that results in the storage
  /// of the captures into capture struct.
  void storeCaptures(OpBuilder &b, Value captureStructArg,
                     ClosureInitData &closureInitData,
                     ArrayRef<Capture> captureMechanisms);
  /// Lift driving function. The loweredClosureType should be a kgen.struct with
  /// the captures.
  Value liftClosure(OpBuilder &b, ClosureInitData &closureInitData,
                    ArrayRef<Capture> captureMechanisms,
                    Type loweredClosureType, Type loweredClosureInstType);
};
} // namespace

namespace llvm {

template <>
struct DenseMapInfo<ClosureLifter::ClosureParentKey> {
  static inline ClosureLifter::ClosureParentKey getEmptyKey() {
    return {SymbolRefAttr(), StringAttr()};
  }
  static inline ClosureLifter::ClosureParentKey getTombstoneKey() {
    auto p = SymbolRefAttr::getFromOpaquePointer(reinterpret_cast<void *>(1));
    auto n = StringAttr::getFromOpaquePointer(reinterpret_cast<void *>(1));
    return {p, n};
  }
  static unsigned getHashValue(const ClosureLifter::ClosureParentKey &k) {
    return ::llvm::hash_combine(k.parent.getAsOpaquePointer(),
                                k.nestedName.getAsOpaquePointer());
  }
  static bool isEqual(const ClosureLifter::ClosureParentKey &a,
                      const ClosureLifter::ClosureParentKey &b) {
    return a.parent == b.parent && a.nestedName == b.nestedName;
  }
};

template <>
struct DenseMapInfo<ClosureLifter::ClosureMethodKey> {
  static inline ClosureLifter::ClosureMethodKey getEmptyKey() {
    return {SymbolRefAttr(), StringAttr(), ClosureMethodAttr()};
  }
  static inline ClosureLifter::ClosureMethodKey getTombstoneKey() {
    auto p = SymbolRefAttr::getFromOpaquePointer(reinterpret_cast<void *>(1));
    auto n = StringAttr::getFromOpaquePointer(reinterpret_cast<void *>(1));
    auto m =
        ClosureMethodAttr::getFromOpaquePointer(reinterpret_cast<void *>(1));
    return {p, n, m};
  }
  static unsigned getHashValue(const ClosureLifter::ClosureMethodKey &k) {
    return ::llvm::hash_combine(k.parent.getAsOpaquePointer(),
                                k.nestedName.getAsOpaquePointer(),
                                k.method.getAsOpaquePointer());
  }
  static bool isEqual(const ClosureLifter::ClosureMethodKey &a,
                      const ClosureLifter::ClosureMethodKey &b) {
    return a.parent == b.parent && a.nestedName == b.nestedName &&
           a.method == b.method;
  }
};

} // namespace llvm

ClosureLifter::ClosureInitData::ClosureInitData(
    llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
    ClosureType closureType, ClosureInitOp closureInit,
    StructGeneratorOp structGeneratorOp, GeneratorOp generator,
    SmallVector<TypedAttr> &&moveSymbols,
    std::optional<SmallVector<TypedAttr>> &&copySymbols)
    : capturedParamDecls(std::move(capturedParamDecls)),
      closureType(closureType), closureInit(closureInit), generator(generator),
      moveSymbols(std::move(moveSymbols)),
      copySymbolsMaybe(std::move(copySymbols)),
      liftedLocation(FusedLoc::get(
          generator->getContext(),
          Location(DebugInfo::extractSourceLoc(closureInit->getLoc())),
          dyn_cast_or_null<DebugInfo::DISubprogramAttr>(
              closureInit.getNestedFnScope().value_or(Attribute())))) {
  // Create the capture struct.
  SmallVector<Type> paramTypes;
  MLIRContext *cxt = generator->getContext();
  for (ParamDeclAttr paramCaptures : getCapturedParamDecls())
    paramTypes.push_back(paramCaptures.getType());

  paramClosureType = ParamClosureType::get(cxt, closureType.getParentSymbol(),
                                           StringAttr::get(cxt, regionName()));
  structGeneratorOp->walk([&](WitnessOp witness) {
    if (auto closureSym = dyn_cast<ClosureSymbolAttr>(witness.getValue())) {
      abstractSymbolMap[closureSym.getMethod().getValue()] = closureSym;
    }
  });
}

ClosureSymbolAttr ClosureLifter::ClosureInitData::closureSymbolForSourceName(
    ClosureMethod method) const {
  auto sym = abstractSymbolMap.find(method);
  if (sym == abstractSymbolMap.end())
    return {};
  return sym->getSecond();
}

/// A closure init op has two possible types: pointer<closure_type> or
/// closure_type. The closure type encodes where the captures are stored, which
/// is necessary for lowering. Extract the closure type from the result type of
/// the closure init.
static ClosureType getClosureType(ClosureInitOp closureInit) {
  if (auto closureTypeAttr =
          closureInit->getAttrOfType<TypeAttr>("closureType")) {
    if (auto closureType = dyn_cast<ClosureType>(closureTypeAttr.getValue()))
      return closureType;
  }

  Type resultType = closureInit->getResultTypes().front();
  ClosureType closureType;
  if (auto ptr = dyn_cast<PointerType>(closureInit->getResultTypes().front()))
    closureType = dyn_cast<ClosureType>(ptr.getElementType());
  else
    closureType = dyn_cast<ClosureType>(resultType);
  assert(closureType && "closure init must be of closure type");
  return closureType;
}

void ClosureLifter::createClosureGenerator(
    OpBuilder &b, Location location, ClosureInitData &closureInitData,
    ClosureMethod method, FunctionType funcType,
    llvm::function_ref<void(GeneratorOp)> populateBody,
    ArrayRef<ArgConvention> argConventions) {
  ClosureSymbolAttr closureAttr =
      closureInitData.closureSymbolForSourceName(method);
  /// If there is no witness for this method then there is no reference to it
  if (!closureAttr)
    return;
  GeneratorOp generator = closureInitData.getGenerator();

  SmallVector<Type> resultTypes;
  resultTypes.push_back(funcType.getResult(0));

  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (Twine(generator.getName()) + "__" + stringifyClosureMethod(method) +
       "__" + closureInitData.regionName())
          .str(),
      symtab, counter));
  b.setInsertionPoint(generator);
  auto closureGenerator =
      GeneratorOp::create(b, location, uniqueName,
                          FuncTypeGeneratorType::get(
                              {}, FunctionType::get(b.getContext(), {}, {})));
  closureGenerator.setSourceNameAttr(
      closureInitData.getClosureType().getName());
  populateBody(closureGenerator);

  closureGenerator.setFuncTypeGenerator(
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          closureInitData.getCapturedParamDecls(), funcType,
          /*argConv=*/argConventions,
          /*effects=*/{},
          /*fnMetadata=*/{}, /*genMetadata=*/{}));
  closureGenerator.setFunctionType(funcType);
  closureGenerator.setInputParams({closureInitData.getCapturedParamDecls()});

  // Map from synthesized function to abstracted symbols.
  SmallVector<TypedAttr> boundParams;
  for (auto capturedParam : closureInitData.getCapturedParamDecls())
    boundParams.push_back(ParamDeclRefAttr::get(capturedParam.getName(),
                                                capturedParam.getType()));
  auto sym = SymbolConstantAttr::get(
      closureGenerator,
      FuncTypeGeneratorType::get({}, funcType, /*argConv=*/argConventions,
                                 /*effects=*/{},
                                 /*fnMetadata=*/{}, /*genMetadata=*/{}),
      boundParams);
  liftedClosureSymbols[{closureAttr.getParentSymbol(),
                        closureAttr.getNestedFuncName(),
                        closureAttr.getMethod()}] = sym;
  symtab.insert(closureGenerator);
}

void ClosureLifter::liftDelFunction(OpBuilder &b, Location loc,
                                    ClosureInitData &closureInitData,
                                    Type loweredClosureType,
                                    ArrayRef<Capture> captureMechanisms) {
  bool isFlattenedSingletonClosure =
      closureInitData.isFlattenedSingletonClosure(captureMechanisms.size());
  Type selfType = closureInitData.selfType(loweredClosureType);
  SmallVector<Type> argTypes;
  argTypes.push_back(selfType);
  FunctionType funcType = FunctionType::get(
      b.getContext(), argTypes, {KGEN::NoneType::get(b.getContext())});
  auto populateBody = [&](GeneratorOp delGenerator) {
    Block &delBlock = delGenerator.getBodyRegion().emplaceBlock();
    for (Type type : argTypes)
      delBlock.addArgument(type, delGenerator.getLoc());
    b.setInsertionPointToStart(&delBlock);
    Value source = delBlock.getArgument(0);
    for (auto [index, capture] : llvm::enumerate(captureMechanisms)) {
      if (capture.delSym.has_value()) {
        Value field = isFlattenedSingletonClosure
                          ? source
                          : KGEN::StructGEPOp::create(b, loc, source, index);
        TypedAttr delSym = *capture.delSym;
        auto sig = cast<FuncTypeGeneratorType>(delSym.getType());
        Type resultType = sig.getBody().getResults().front();
        KGEN::CallParamOp::create(b, loc, resultType, delSym,
                                  ValueRange(field));
      }
    }
    auto noneAttr = KGEN::ParamConstantOp::create(
        b, loc, KGEN::NoneAttr::get(b.getContext()));
    KGEN::ReturnOp::create(b, loc, noneAttr->getResults().front());
  };
  createClosureGenerator(b, loc, closureInitData, ClosureMethod::DEL, funcType,
                         populateBody, {ArgConvention::DeinitMem});
}

static void emitCopyMoveCall(mlir::OpBuilder &b, Location location,
                             TypedAttr symbol, Value original, Value slot) {
  auto sig = cast<FuncTypeGeneratorType>(symbol.getType());
  Type resultType = sig.getBody().getResults().front();
  if (sig.getBody().getArguments().size() == 2) {
    SmallVector<Value> values = {original, slot};
    KGEN::CallParamOp::create(b, location, resultType, symbol,
                              ValueRange(values));
  } else {
    auto callOp = KGEN::CallParamOp::create(b, location, resultType, symbol,
                                            ValueRange(original));
    POP::StoreOp::create(b, location, callOp->getResults().front(), slot);
  }
}

void ClosureLifter::liftMoveOrCopyFunction(OpBuilder &b, Location loc,
                                           ClosureInitData &closureInitData,
                                           Type loweredClosureType,
                                           ArrayRef<Capture> captureMechanisms,
                                           bool isMove) {
  bool isFlattenedSingletonClosure =
      closureInitData.isFlattenedSingletonClosure(captureMechanisms.size());
  Type selfType = closureInitData.selfType(loweredClosureType);
  SmallVector<Type> argTypes{selfType, selfType};
  FunctionType funcType = FunctionType::get(
      b.getContext(), argTypes, {KGEN::NoneType::get(b.getContext())});
  auto populateBody = [&](GeneratorOp generator) {
    Block &moveBlock = generator.getBodyRegion().emplaceBlock();
    for (Type type : argTypes)
      moveBlock.addArgument(type, generator.getLoc());
    b.setInsertionPointToStart(&moveBlock);
    Value source = moveBlock.getArgument(0);
    Value target = moveBlock.getArgument(1);
    unsigned symIndex = 0;
    for (auto [index, capture] : llvm::enumerate(captureMechanisms)) {
      Value targetField =
          isFlattenedSingletonClosure
              ? target
              : KGEN::StructGEPOp::create(b, loc, target, index);
      Value sourceField =
          isFlattenedSingletonClosure
              ? source
              : KGEN::StructGEPOp::create(b, loc, source, index);
      if (!capture.moveOrCopySym.has_value()) {
        POP::StoreOp::create(b, loc, POP::LoadOp::create(b, loc, sourceField),
                             targetField);
      } else {
        TypedAttr symbol = isMove
                               ? closureInitData.getMoveSymbols()[symIndex++]
                               : closureInitData.getCopySymbols()[symIndex++];
        emitCopyMoveCall(b, loc, symbol, sourceField, targetField);
      }
    }
    auto noneAttr = KGEN::ParamConstantOp::create(
        b, loc, KGEN::NoneAttr::get(b.getContext()));
    KGEN::ReturnOp::create(b, loc, noneAttr->getResults().front());
  };
  createClosureGenerator(
      b, loc, closureInitData,
      isMove ? ClosureMethod::MOVE : ClosureMethod::COPY, funcType,
      populateBody,
      {isMove ? ArgConvention::DeinitMem : ArgConvention::ReadMem,
       ArgConvention::ByRefResult});
}

void ClosureLifter::storeCaptures(OpBuilder &b, Value captureStruct,
                                  ClosureInitData &data,
                                  ArrayRef<Capture> captureMechanisms) {
  bool isFlattenedSingletonClosure =
      data.isFlattenedSingletonClosure(captureMechanisms.size());
  b.setInsertionPoint(data.getClosureInit());
  Location location = data.getClosureInit()->getLoc();
  for (auto [index, captureMechanism] : llvm::enumerate(captureMechanisms)) {
    Value slot = isFlattenedSingletonClosure
                     ? captureStruct
                     : StructGEPOp::create(b, location, captureStruct, index);
    if (captureMechanism.moveOrCopySym.has_value()) {
      TypedAttr symbol = *captureMechanism.moveOrCopySym;
      emitCopyMoveCall(b, location, symbol, captureMechanism.origin, slot);
    } else {
      POP::StoreOp::create(b, location, captureMechanism.origin, slot);
    }
  }
}

Value ClosureLifter::liftClosure(OpBuilder &b, ClosureInitData &closureInitData,
                                 ArrayRef<Capture> captureMechanisms,
                                 Type loweredClosureType,
                                 Type loweredClosureInstType) {
  Location loc = closureInitData.getClosureInit().getLoc();
  Type selfType = closureInitData.selfType(loweredClosureType);
  // The closure call implementation is now promoted in the parser.
  // This pass only synthesizes memory-management methods and capture storage.

  // Instantiate capture struct.
  b.setInsertionPoint(closureInitData.getClosureInit());
  Value captureStruct =
      closureInitData.isEscaping()
          ? allocateHeapMemory(cast<PointerType>(selfType), b, loc)
          : POP::StackAllocationOp::create(b, loc,
                                           /*markedLifetimes=*/true,
                                           PointerType::get(loweredClosureType))
                .getResult();
  storeCaptures(b, captureStruct, closureInitData, captureMechanisms);
  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;

  closureTypeToStructInstTypes[closureInitData.getClosureType()] =
      loweredClosureInstType;

  return captureStruct;
}

Value ClosureLifter::liftRegPassableClosure(OpBuilder &b,
                                            ClosureInitData &closureInitData,
                                            ArrayRef<Capture> captureMechanisms,
                                            Type loweredClosureType,
                                            Type loweredClosureInstType) {
  Value captureStruct = liftClosure(b, closureInitData, captureMechanisms,
                                    loweredClosureType, loweredClosureInstType);
  if (!captureStruct)
    return {};
  return POP::LoadOp::create(b, closureInitData.getClosureInit()->getLoc(),
                             captureStruct);
}

Value ClosureLifter::liftNonRegPassableClosure(
    OpBuilder &b, ClosureInitData &closureInitData,
    ArrayRef<Capture> captureMechanisms, Type loweredClosureType,
    Type loweredClosureInstType) {
  Value captureStruct = liftClosure(b, closureInitData, captureMechanisms,
                                    loweredClosureType, loweredClosureInstType);
  Location loc = stripParameterRefsFromLoc(closureInitData.getLiftedLocation());
  if (!captureStruct)
    return {};
  liftMoveOrCopyFunction(b, loc, closureInitData, loweredClosureType,
                         captureMechanisms, /*isMove=*/true);
  if (closureInitData.isCopyable())
    liftMoveOrCopyFunction(b, loc, closureInitData, loweredClosureType,
                           captureMechanisms, /*isMove=*/false);
  liftDelFunction(b, loc, closureInitData, loweredClosureType,
                  captureMechanisms);
  return captureStruct;
}

Value ClosureLifter::liftThinClosure(OpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     bool isRegisterPassable,
                                     Type loweredClosureType,
                                     Type loweredClosureInstType) {
  // The closure call implementation is now promoted in the parser.
  // TODO: create thunks for register passable closures (MOCO-2242).
  if (!isRegisterPassable) {
    Location liftedLoc =
        stripParameterRefsFromLoc(closureInitData.getLiftedLocation());
    liftMoveOrCopyFunction(b, liftedLoc, closureInitData, loweredClosureType,
                           {}, /*isMove=*/true);
    if (closureInitData.isCopyable())
      liftMoveOrCopyFunction(b, liftedLoc, closureInitData, loweredClosureType,
                             {}, /*isMove=*/false);
    liftDelFunction(b, liftedLoc, closureInitData, loweredClosureType, {});
  }

  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;
  closureTypeToStructInstTypes[closureInitData.getClosureType()] =
      loweredClosureInstType;

  b.setInsertionPoint(closureInitData.getClosureInit());
  Location loc = closureInitData.getClosureInit()->getLoc();
  if (isRegisterPassable)
    return StructCreateOp::create(b, loc, cast<StructType>(loweredClosureType),
                                  ValueRange())
        .getResult();
  return POP::StackAllocationOp::create(b, loc,
                                        /*markedLifetimes=*/true,
                                        PointerType::get(loweredClosureType))
      .getResult();
}

LogicalResult
ClosureLifter::liftClosureInit(ClosureInitOp closureInit, GeneratorOp generator,
                               StructGeneratorOp structGeneratorOp) {
  OpBuilder b(closureInit.getContext());
  ClosureType closureType = getClosureType(closureInit);
  ClosureMemoryKind memoryKind = closureType.getClosureMemoryKind();
  b.setInsertionPoint(closureInit);
  DenseMap<Value, Attribute> captureToSymbol;
  for (auto [capture, symbol] :
       llvm::zip(closureInit.getCaptures(),
                 closureInit.getMoveOrCopyCaptureSymbols()))
    captureToSymbol[capture] = symbol;

  // Enforce that all captures specify a capture convention.
  SmallVector<Type> fieldTypes;
  SmallVector<Capture> captureMechanisms;

  // Build capture mechanisms and copy/move symbols.
  SmallVector<TypedAttr> moveSymbols;
  SmallVector<TypedAttr> copySymbols;
  bool allCopySymbolsAvailable = true;
  for (Value capture : closureInit.getCaptures()) {
    auto ptr = captureToSymbol.find(capture);
    assert(ptr != captureToSymbol.end() && "capture must be in capture list");
    if (auto triple = dyn_cast<MemSymbolTripleAttr>(ptr->second)) {
      TypedAttr symbol =
          triple.getIsMove() ? triple.getMove() : triple.getCopy();
      TypedAttr moveSymbol = triple.getMove();
      TypedAttr copySymbol = triple.getCopy();
      if (moveSymbol)
        moveSymbols.push_back(moveSymbol);
      else if (copySymbol)
        moveSymbols.push_back(copySymbol);
      else
        llvm_unreachable("cannot capture by move or copy and not include a "
                         "move or copy symbol");
      if (copySymbol && allCopySymbolsAvailable)
        copySymbols.push_back(copySymbol);
      else
        allCopySymbolsAvailable = false;
      TypedAttr del = triple.getDel();
      Type capturingType =
          cast<PointerType>(capture.getType()).getElementType();
      fieldTypes.push_back(capturingType);
      captureMechanisms.push_back({symbol, del, capture});
      continue;
    }
    fieldTypes.push_back(capture.getType());
    captureMechanisms.push_back({{}, {}, capture});
  }
  std::optional<SmallVector<TypedAttr>> copiesMaybe;
  if (allCopySymbolsAvailable)
    copiesMaybe = copySymbols;
  llvm::SetVector<ParamDeclAttr> capturedParamDecls;
  auto [closureInitData, loweredClosureInstType, loweredClosureType] =
      [&]() -> std::tuple<ClosureInitData, StructInstanceType, Type> {
    // If the type value attribute is set, use its genref bindings as captured
    // params and use the struct generator's value domain type as the struct
    // instance type.
    if (auto typeValueData =
            getTypeValuePathData(closureInit, structGeneratorOp)) {
      auto [typeGeneratorRef, structInstanceType] = *typeValueData;
      SmallVector<Type> structFieldTypes;
      structFieldTypes.reserve(structInstanceType.getFields().size());
      for (StructDefFieldAttr field : structInstanceType.getFields()) {
        TypedAttr fieldTypeValue = field.getTypeValue();
        if (auto fieldTypeParam = dyn_cast<TypeParamAttr>(fieldTypeValue))
          structFieldTypes.push_back(fieldTypeParam.getMlirType());
        else if (isa<ParamDeclRefAttr>(fieldTypeValue))
          structFieldTypes.push_back(ParamType::get(fieldTypeValue));
        else
          structFieldTypes.push_back(fieldTypeValue.getType());
      }
      Type structType = StructType::get(
          b.getContext(), structFieldTypes,
          cast<BoolAttr>(structInstanceType.getIsMemoryOnly()).getValue());
      if (structFieldTypes.size() == 1 &&
          !cast<BoolAttr>(structInstanceType.getIsMemoryOnly()).getValue())
        structType = structFieldTypes.front();
      for (TypedAttr binding : typeGeneratorRef.getParamValues())
        if (auto ref = dyn_cast<ParamDeclRefAttr>(binding))
          capturedParamDecls.insert(
              ParamDeclAttr::get(ref.getName(), ref.getType()));

      return {ClosureInitData(std::move(capturedParamDecls), closureType,
                              closureInit, structGeneratorOp, generator,
                              std::move(moveSymbols), std::move(copiesMaybe)),
              structInstanceType, structType};
    }
    // Otherwise, fallback to computing the struct type from the metadata
    DenseMap<Value, std::pair<StringAttr, TypedAttr>> captureToNameAndType;
    ArrayAttr captureNamesArr = closureInit.getCaptureNames();
    ArrayAttr captureTypesArr = closureInit.getCaptureTypes();
    for (auto [val, nameAttr, typeAttr] : llvm::zip(
             closureInit.getCaptures(), captureNamesArr, captureTypesArr)) {
      captureToNameAndType[val] = {cast<StringAttr>(nameAttr),
                                   cast<TypedAttr>(typeAttr)};
    }

    MLIRContext *ctx = b.getContext();
    SmallVector<StructDefFieldAttr> fieldDecls;
    for (auto &capture : captureMechanisms) {
      if (auto it = captureToNameAndType.find(capture.origin);
          it != captureToNameAndType.end()) {
        StringAttr name = it->second.first;
        TypedAttr typeValue = it->second.second;
        fieldDecls.push_back(StructDefFieldAttr::get(name, typeValue));
      }
    }
    bool isMemoryOnly = memoryKind != ClosureMemoryKind::TRIVIAL &&
                        memoryKind != ClosureMemoryKind::REGISTER_PASSABLE;

    auto structInstanceType = StructInstanceType::get(
        structGeneratorOp.getSymNameAttr(),
        /*paramNames=*/{},
        /*paramValues=*/{}, fieldDecls, BoolAttr::get(ctx, isMemoryOnly));
    Type mlirType = StructType::get(b.getContext(), fieldTypes, isMemoryOnly);
    if (fieldTypes.size() == 1 && !isMemoryOnly)
      mlirType = fieldTypes.front();
    for (ParamDeclAttr input : closureInit.getInputParams())
      capturedParamDecls.insert(input);
    if (auto hoistedCaptures = closureInit.getHoistedCaptures())
      for (ParamDeclAttr hoisted : *hoistedCaptures)
        capturedParamDecls.insert(hoisted);

    return {ClosureInitData(std::move(capturedParamDecls), closureType,
                            closureInit, structGeneratorOp, generator,
                            std::move(moveSymbols), std::move(copiesMaybe)),
            structInstanceType, mlirType};
  }();
  bool isThin = fieldTypes.empty();

  // Replace runtime abstractions.
  Value replacement;
  if (isThin) {
    replacement = liftThinClosure(b, closureInitData,
                                  /*isRegisterPassable=*/memoryKind ==
                                      ClosureMemoryKind::TRIVIAL,
                                  loweredClosureType, loweredClosureInstType);
  } else {
    switch (memoryKind) {
    case ClosureMemoryKind::TRIVIAL:
      replacement =
          liftRegPassableClosure(b, closureInitData, captureMechanisms,
                                 loweredClosureType, loweredClosureInstType);
      break;
    case ClosureMemoryKind::REGISTER_PASSABLE:
    case ClosureMemoryKind::ESCAPING:
    case ClosureMemoryKind::NONESCAPING:
      replacement =
          liftNonRegPassableClosure(b, closureInitData, captureMechanisms,
                                    loweredClosureType, loweredClosureInstType);
      break;
    }
  }
  if (!replacement)
    return failure();
  closureInit.getResult().replaceAllUsesWith(replacement);
  closureInit.erase();
  ClosureParentKey key{closureInitData.getClosureType().getParentSymbol(),
                       closureInitData.getClosureType().getName()};
  paramCaptureToStructAttr[key] = std::move(capturedParamDecls);
  return success();
}

static StringAttr getFullName(ClosureType closureType) {
  MLIRContext *ctx = closureType.getContext();
  StringRef parentName = closureType.getParentSymbol().getRootReference();
  StringRef closureNameRef = closureType.getName();
  SmallString<64> fullName;
  fullName.reserve(parentName.size() + 1 + closureNameRef.size());

  fullName += parentName;
  fullName += "::";
  fullName += closureNameRef;

  return StringAttr::get(ctx, fullName);
}

static LogicalResult liftClosureInit(ModuleOp theModule, ClosureLifter &lifter,
                                     ClosureInitOp closureInit) {
  GeneratorOp parent = closureInit->getParentOfType<GeneratorOp>();
  assert(parent && "closure init should be nested within a generator");

  ClosureType closureType = getClosureType(closureInit);
  StringAttr symbol = getFullName(closureType);
  StructGeneratorOp structGeneratorOp =
      lifter.symtab.lookup<StructGeneratorOp>(symbol);
  if (!structGeneratorOp) {
    mlir::emitError(theModule.getLoc())
        << "missing struct generator op for closure " << closureType.getName();
    return failure();
  }
  if (failed(lifter.liftClosureInit(closureInit, parent, structGeneratorOp)))
    return failure();

  ClosureLifter::ClosureParentKey key{closureType.getParentSymbol(),
                                      closureType.getName()};
  auto captures = lifter.paramCaptureToStructAttr.find(key);
  assert(captures != lifter.paramCaptureToStructAttr.end() &&
         "lifting must populate captured parameter set");
  SmallVector<ParamDeclAttr> inputParams;
  llvm::append_range(inputParams, captures->second);
  structGeneratorOp.setInputParams(inputParams);

  auto instTypeIt = lifter.closureTypeToStructInstTypes.find(closureType);
  if (instTypeIt != lifter.closureTypeToStructInstTypes.end()) {
    structGeneratorOp.setValueDomainType(instTypeIt->second);
    lifter.closureTypeToStructGen.insert({closureType, structGeneratorOp});
  }
  return success();
}

// lift closures and replace closure.init
void OutlineClosuresNewPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  ClosureLifter lifter(symtab, debugBuild);
  // Lift every closure init in the module. Each successful lift also lowers
  // the matching struct generator.
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    SmallVector<ClosureInitOp> closuresToLift;
    generator.walk<mlir::WalkOrder::PostOrder>([&](ClosureInitOp closureInit) {
      closuresToLift.push_back(closureInit);
    });
    for (ClosureInitOp closureInit : closuresToLift) {
      if (failed(liftClosureInit(theModule, lifter, closureInit)))
        return signalPassFailure();
    }
  }

  mlir::AttrTypeReplacer closureTypeReplacer;
  closureTypeReplacer.addReplacement([&](ClosureType closureType) -> Type {
    auto it = lifter.closureTypeToStructTypes.find(closureType);
    if (it != lifter.closureTypeToStructTypes.end())
      return it->second;
    mlir::emitError(theModule.getLoc())
        << "no type found for closure type " << closureType;
    return closureType;
  });

  closureTypeReplacer.addReplacement([&](TypeParamAttr attr) -> Attribute {
    if (auto closureType = dyn_cast<ClosureType>(attr.getTypeValue())) {
      auto it = lifter.closureTypeToStructTypes.find(closureType);
      auto genIt = lifter.closureTypeToStructGen.find(closureType);
      if (genIt != lifter.closureTypeToStructGen.end() &&
          it != lifter.closureTypeToStructTypes.end()) {
        auto genref = TypeGeneratorRefAttr::get(
            SymbolRefAttr::get(genIt->second.getSymNameAttr()),
            genIt->second.getMetaType());
        auto result = TypeParamAttr::get(TypeValueType::get(genref), it->second,
                                         attr.getType());
        return result;
      } else {
        mlir::emitError(theModule.getLoc())
            << "no symbol or generator found " << closureType;
      }
    }
    return attr;
  });

  closureTypeReplacer.addReplacement(
      [&](ClosureSymbolAttr symbol) -> Attribute {
        ClosureLifter::ClosureMethodKey key{symbol.getParentSymbol(),
                                            symbol.getNestedFuncName(),
                                            symbol.getMethod()};
        auto it = lifter.liftedClosureSymbols.find(key);

        if (it != lifter.liftedClosureSymbols.end())
          return it->second;
        mlir::emitError(theModule.getLoc())
            << "no symbol found for closure symbol " << symbol;
        return symbol;
      });

  // TODO: Replace this with a proper abstraction
  closureTypeReplacer.addReplacement([&](TypeGeneratorRefAttr typeValueType) {
    ClosureAttr closureAttr;
    for (auto param : typeValueType.getParamValues()) {
      if (auto closure = dyn_cast<ClosureAttr>(param)) {
        closureAttr = closure;
        break;
      }
    }
    if (!closureAttr)
      return typeValueType;
    ParamClosureType paramClosureType = closureAttr.getType();
    ClosureLifter::ClosureParentKey key{paramClosureType.getParentSymbol(),
                                        paramClosureType.getName()};
    auto captures = lifter.paramCaptureToStructAttr.find(key);
    if (captures == lifter.paramCaptureToStructAttr.end()) {
      mlir::emitError(theModule.getLoc())
          << "no captures found for closure symbol " << closureAttr;
    }
    SmallVector<TypedAttr> boundParams;
    for (auto p : captures->second)
      boundParams.push_back(ParamDeclRefAttr::get(p.getName(), p.getType()));
    return TypeGeneratorRefAttr::get(theModule.getContext(),
                                     typeValueType.getSymbol(), boundParams,
                                     typeValueType.getType());
  });

  for (Operation &operation : *theModule.getBody()) {
    if (isa<GeneratorOp, StructGeneratorOp>(operation))
      closureTypeReplacer.recursivelyReplaceElementsIn(&operation, true, true,
                                                       true);
  }
}
