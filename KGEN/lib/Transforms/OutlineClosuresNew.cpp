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

static TypeGeneratorRefAttr getTypeGeneratorRef(ClosureInitOp closureInit) {
  auto typeValue = closureInit->getAttrOfType<TypedAttr>("typeValue");
  if (!typeValue)
    return {};
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return {};
  auto typeValueType = dyn_cast<TypeValueType>(typeParam.getTypeValue());
  if (!typeValueType)
    return {};
  return dyn_cast<TypeGeneratorRefAttr>(typeValueType.getTypeValue());
}

static std::optional<std::pair<TypeGeneratorRefAttr, StructInstanceType>>
getTypeValuePathData(ClosureInitOp closureInit,
                     StructGeneratorOp structGeneratorOp) {
  TypeGeneratorRefAttr typeGeneratorRef = getTypeGeneratorRef(closureInit);
  if (!typeGeneratorRef)
    return std::nullopt;
  auto structInstanceType =
      dyn_cast<StructInstanceType>(structGeneratorOp.getValueDomainType());
  if (!structInstanceType)
    return std::nullopt;
  return std::make_pair(typeGeneratorRef, structInstanceType);
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
                    GeneratorOp generator);
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

  private:
    llvm::SetVector<ParamDeclAttr> capturedParamDecls;
    ClosureType closureType;
    ClosureInitOp closureInit;
    GeneratorOp generator;
    ParamClosureType paramClosureType;
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

} // namespace llvm

ClosureLifter::ClosureInitData::ClosureInitData(
    llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
    ClosureType closureType, ClosureInitOp closureInit, GeneratorOp generator)
    : capturedParamDecls(std::move(capturedParamDecls)),
      closureType(closureType), closureInit(closureInit), generator(generator) {
  MLIRContext *cxt = generator->getContext();
  paramClosureType = ParamClosureType::get(cxt, closureType.getParentSymbol(),
                                           StringAttr::get(cxt, regionName()));
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
  if (!captureStruct)
    return {};
  return captureStruct;
}

Value ClosureLifter::liftThinClosure(OpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     bool isRegisterPassable,
                                     Type loweredClosureType,
                                     Type loweredClosureInstType) {
  // TODO: create thunks for register passable closures (MOCO-2242).
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

  // Build capture mechanisms.
  for (Value capture : closureInit.getCaptures()) {
    auto ptr = captureToSymbol.find(capture);
    assert(ptr != captureToSymbol.end() && "capture must be in capture list");
    if (auto triple = dyn_cast<MemSymbolTripleAttr>(ptr->second)) {
      TypedAttr symbol =
          triple.getIsMove() ? triple.getMove() : triple.getCopy();
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
                              closureInit, generator),
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
                            closureInit, generator),
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

static LogicalResult liftClosureInit(ModuleOp theModule, ClosureLifter &lifter,
                                     ClosureInitOp closureInit) {
  GeneratorOp parent = closureInit->getParentOfType<GeneratorOp>();
  assert(parent && "closure init should be nested within a generator");

  ClosureType closureType = getClosureType(closureInit);
  StructGeneratorOp structGeneratorOp;
  TypeGeneratorRefAttr typeGeneratorRef = getTypeGeneratorRef(closureInit);
  if (typeGeneratorRef) {
    structGeneratorOp = lifter.symtab.lookup<StructGeneratorOp>(
        typeGeneratorRef.getSymbol().getRootReference());
  }
  if (!structGeneratorOp) {
    mlir::emitError(theModule.getLoc())
        << "missing storage struct generator op for closure "
        << closureType.getName();
    return failure();
  }
  if (failed(lifter.liftClosureInit(closureInit, parent, structGeneratorOp)))
    return failure();

  ClosureLifter::ClosureParentKey key{closureType.getParentSymbol(),
                                      closureType.getName()};
  auto captures = lifter.paramCaptureToStructAttr.find(key);
  assert(captures != lifter.paramCaptureToStructAttr.end() &&
         "lifting must populate captured parameter set");

  auto instTypeIt = lifter.closureTypeToStructInstTypes.find(closureType);
  if (instTypeIt != lifter.closureTypeToStructInstTypes.end()) {
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
