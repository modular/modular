//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"

#include "mlir/IR/BuiltinOps.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "outline-closures-new"

namespace M::KGEN {
#define GEN_PASS_DEF_OUTLINECLOSURESNEW
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct OutlineClosuresNewPass
    : M::KGEN::impl::OutlineClosuresNewBase<OutlineClosuresNewPass> {
  using OutlineClosuresNewBase::OutlineClosuresNewBase;

  void runOnOperation() override;
};
} // namespace

namespace {
struct Capture {
  /// the symbol of the copy/move constructor, if applicable
  std::optional<SymbolConstantAttr> moveOrCopySym;
  /// the source of the capture. This is the value used to create a copy if the
  /// symbol is nonnull.
  Value origin;
};
} // namespace

static Value allocateHeapMemory(PointerType ptrType, ImplicitLocOpBuilder &b) {
  TypedAttr elementType = TypeParamAttr::get(
      ptrType.getElementType(), TypeType::get(ptrType.getContext()));
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value sizeOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return b.create<POP::AlignedAllocOp>(ptrType, ValueRange{alignOf, sizeOf});
}

namespace {
/// The ClosureLifter is responsible for
/// (a) lifting a closure init into a top level function + capture struct and
/// (b) storing metadata necessary to replace references to the closure.
struct ClosureLifter {
  ClosureLifter(SymbolTable &symtab) : counter(0), symtab(symtab) {}
  /// Given components of the lifted function, generate a closure symbol, which
  /// is an abstraction of a symbol used to reference functions that do not yet
  /// exist.
  ClosureSymbolAttr createClosureSymbolAttr(GeneratorOp parent, StringRef name,
                                            ClosureMethod method,
                                            ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resultTypes,
                                            ClosureType closureType);
  /// Given a closure init op, generate functions for call, copy, move, and del
  /// + struct instance to store captures.
  LogicalResult liftClosureInit(ClosureInitOp closureInit,
                                GeneratorOp generator);

  /// Symbol name uniquer requires a counter.
  unsigned counter;
  /// The symbol table of the module.
  SymbolTable &symtab;
  /// Pair a closure symbol with the symbol of the lifted function so that the
  /// closure symbols can be replaced.
  DenseMap<ClosureSymbolAttr, SymbolConstantAttr> liftedClosureSymbols;
  /// Pair the closure type with the struct type of the generated capture struct
  /// so that the closure types can be replaced.
  DenseMap<ClosureType, Type> closureTypeToStructTypes;
  struct ClosureInitData {
    ClosureInitData(ClosureType closureType, ClosureInitOp closureInit,
                    GeneratorOp generator)
        : closureType(closureType), closureInit(closureInit),
          generator(generator) {}
    Type selfType(Type loweredClosureType) const {
      return closureType.getClosureMemoryKind() ==
                     ClosureMemoryKind::REGISTER_PASSABLE
                 ? loweredClosureType
                 : PointerType::get(loweredClosureType);
    }
    Region &region() { return closureInit->getRegions().front(); }
    StringRef regionName() const { return closureType.getName(); }
    ArrayRef<Type> results() {
      return closureInit.getFunctionType().getResults();
    }
    GeneratorOp getGenerator() const { return generator; }
    ClosureType getClosureType() const { return closureType; }
    ClosureInitOp getClosureInit() const { return closureInit; }
    bool isEscaping() const {
      return closureType.getClosureMemoryKind() == ClosureMemoryKind::ESCAPING;
    }

  private:
    ClosureType closureType;
    ClosureInitOp closureInit;
    GeneratorOp generator;
  };

private:
  /// Lift a register passable closure. The characterization is in the lifted
  /// signature: a register passable closure's lifted call function has an
  /// implicit self argument of struct type.
  Value liftRegPassableClosure(ImplicitLocOpBuilder &b, ClosureInitData &data,
                               ArrayRef<Capture> captureMechanisms,
                               Type loweredClosureType);
  /// Lift a closure with no captures. We can skip the loading/storing of
  /// captures and the self type is none or opaque pointer, depending on the
  /// register passable flag.
  Value liftThinClosure(ImplicitLocOpBuilder &b, ClosureInitData &data,
                        bool isRegisterPassable);
  /// Lift a non-register passable closure. The characterization is in the
  /// lifted signature: a non-register passable closure's lifted call function
  /// has an implicit self argument of pointer type.
  Value liftNonRegPassableClosure(ImplicitLocOpBuilder &b,
                                  ClosureInitData &closureInitData,
                                  ArrayRef<Capture> captureMechanisms,
                                  Type loweredClosureType);
  /// Given closure metadata, lift the region of the closure init into a top
  /// level function.
  void liftCallFunction(ImplicitLocOpBuilder &b, ClosureInitData &data);
  /// Given closure metadata the captures, emit code that results in the storage
  /// of the captures into capture struct.
  void storeCaptures(ImplicitLocOpBuilder &b, Value captureStructArg,
                     ClosureInitData &closureInitData,
                     ArrayRef<Capture> captureMechanisms);
  /// Lift driving function. The loweredClosureType should be a kgen.struct with
  /// the captures and the replacement function is meant to emit the IR
  /// necessary for extracting the capture value out of the closure struct
  /// instance.
  Value liftClosure(ImplicitLocOpBuilder &b, ClosureInitData &closureInitData,
                    ArrayRef<Capture> captureMechanisms,
                    Type loweredClosureType,
                    function_ref<Value(Capture, int, Value)> replacementFn);
};
} // namespace

ClosureSymbolAttr ClosureLifter::createClosureSymbolAttr(
    GeneratorOp parent, StringRef name, ClosureMethod method,
    ArrayRef<Type> argTypes, ArrayRef<Type> resultTypes,
    ClosureType closureType) {
  MLIRContext *cxt = parent->getContext();
  SmallVector<Type> originalArgTypes;
  Type selfType =
      closureType.getClosureMemoryKind() == ClosureMemoryKind::REGISTER_PASSABLE
          ? (Type)closureType
          : PointerType::get(closureType);
  originalArgTypes.push_back(selfType);
  llvm::append_range(originalArgTypes, argTypes);
  FunctionType originalFuncType =
      FunctionType::get(cxt, originalArgTypes, resultTypes);
  M::KGEN::FuncTypeGeneratorType originalfuncGenType =
      M::KGEN::FuncTypeGeneratorType::get({}, originalFuncType);
  // TODO: Add parameters MOCO-1740
  auto closureAttr = ClosureSymbolAttr::get(
      cxt, SymbolRefAttr::get(parent.getSymNameAttr()),
      StringAttr::get(cxt, name), ClosureMethodAttr::get(cxt, method), {},
      originalfuncGenType);
  return closureAttr;
}

/// A closure init op has two possible types: pointer<closure_type> or
/// closure_type. The closure type encodes where the captures are stored, which
/// is necessary for lowering. Extract the closure type from the result type of
/// the closure init.
static ClosureType getClosureType(ClosureInitOp closureInit) {
  Type resultType = closureInit->getResultTypes().front();
  ClosureType closureType;
  if (auto ptr = dyn_cast<PointerType>(closureInit->getResultTypes().front()))
    closureType = dyn_cast<ClosureType>(ptr.getElementType());
  else
    closureType = dyn_cast<ClosureType>(resultType);
  assert(closureType && "closure init must be of closure type");
  return closureType;
}

void ClosureLifter::liftCallFunction(ImplicitLocOpBuilder &b,
                                     ClosureInitData &closureInitData) {
  Region &region = closureInitData.region();
  GeneratorOp generator = closureInitData.getGenerator();
  SmallVector<Type> argTypes;
  llvm::append_range(argTypes, region.getArgumentTypes());
  b.setInsertionPoint(generator);
  FunctionType funcType =
      FunctionType::get(b.getContext(), argTypes, closureInitData.results());
  M::KGEN::FuncTypeGeneratorType funcGenType =
      M::KGEN::FuncTypeGeneratorType::get({}, funcType);
  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + "_" + closureInitData.regionName()).str(), symtab,
      counter));
  auto liftedWrapper = b.create<GeneratorOp>(uniqueName, funcGenType);
  symtab.insert(liftedWrapper);
  auto sym = SymbolConstantAttr::get(liftedWrapper);
  Region &body = liftedWrapper.getBodyRegion();
  body.takeBody(region);

  // The closure symbol does not have the implicit argument; remove it
  argTypes.erase(argTypes.begin());
  ClosureSymbolAttr closureAttr = createClosureSymbolAttr(
      closureInitData.getGenerator(), closureInitData.regionName(),
      ClosureMethod::CALL, argTypes, closureInitData.results(),
      closureInitData.getClosureType());
  liftedClosureSymbols[closureAttr] = sym;
}

void ClosureLifter::storeCaptures(ImplicitLocOpBuilder &b, Value captureStruct,
                                  ClosureInitData &data,
                                  ArrayRef<Capture> captureMechanisms) {
  b.setInsertionPoint(data.getClosureInit());
  for (auto [index, captureMechanism] : llvm::enumerate(captureMechanisms)) {
    auto slot = b.create<StructGEPOp>(captureStruct, index);
    if (captureMechanism.moveOrCopySym.has_value()) {
      SymbolConstantAttr symbol = *captureMechanism.moveOrCopySym;
      StringRef name = symbol.getSymbol().getRootReference();
      Operation *op = symtab.lookup(name);
      GeneratorOp function = cast<GeneratorOp>(op);
      SmallVector<Value> values = {slot, captureMechanism.origin};
      b.create<KGEN::CallOp>(function.getFunctionType().getResults(), symbol,
                             ValueRange(values));
    } else {
      b.create<POP::StoreOp>(captureMechanism.origin, slot);
    }
  }
}

Value ClosureLifter::liftClosure(
    ImplicitLocOpBuilder &b, ClosureInitData &closureInitData,
    ArrayRef<Capture> captureMechanisms, Type loweredClosureType,
    function_ref<Value(Capture, int, Value)> replacementFn) {
  Region &region = closureInitData.region();
  // Outline Closure.
  Type selfType = closureInitData.selfType(loweredClosureType);
  Value captureStructArg =
      region.insertArgument((unsigned)0, selfType, region.getLoc());
  b.setInsertionPointToStart(&region.front());
  for (auto [index, capture] : llvm::enumerate(captureMechanisms))
    replaceAllUsesInRegionWith(capture.origin,
                               replacementFn(capture, index, captureStructArg),
                               region);
  liftCallFunction(b, closureInitData);

  // Instantiate capture struct.
  b.setInsertionPoint(closureInitData.getClosureInit());
  Value captureStruct =
      closureInitData.isEscaping()
          ? allocateHeapMemory(cast<PointerType>(selfType), b)
          : b.create<POP::StackAllocationOp>(
                 /*markedLifetimes=*/true, PointerType::get(loweredClosureType))
                .getResult();
  storeCaptures(b, captureStruct, closureInitData, captureMechanisms);
  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;
  return captureStruct;
}

Value ClosureLifter::liftRegPassableClosure(ImplicitLocOpBuilder &b,
                                            ClosureInitData &closureInitData,
                                            ArrayRef<Capture> captureMechanisms,
                                            Type loweredClosureType) {
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    return b.create<KGEN::StructExtractOp>(captureStructArg, index)
        ->getResults()
        .front();
  };
  Value captureStruct = liftClosure(b, closureInitData, captureMechanisms,
                                    loweredClosureType, replacementFn);
  return b.create<POP::LoadOp>(captureStruct);
}

Value ClosureLifter::liftNonRegPassableClosure(
    ImplicitLocOpBuilder &b, ClosureInitData &closureInitData,
    ArrayRef<Capture> captureMechanisms, Type loweredClosureType) {
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    Value replacement = b.create<KGEN::StructGEPOp>(captureStructArg, index);
    if (!capture.moveOrCopySym.has_value())
      replacement = b.create<POP::LoadOp>(replacement);
    return replacement;
  };
  Value captureStruct = liftClosure(b, closureInitData, captureMechanisms,
                                    loweredClosureType, replacementFn);
  return captureStruct;
}

Value ClosureLifter::liftThinClosure(ImplicitLocOpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     bool isRegisterPassable) {
  Type loweredClosureType = KGEN::NoneType::get(b.getContext());
  Type selfType = isRegisterPassable ? loweredClosureType
                                     : PointerType::get(loweredClosureType);
  Region &region = closureInitData.region();
  region.insertArgument((unsigned)0, selfType, region.getLoc());
  liftCallFunction(b, closureInitData);
  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;
  b.setInsertionPoint(closureInitData.getClosureInit());
  return isRegisterPassable
             ? b.create<ParamConstantOp>(NoneAttr::get(b.getContext()))
                   .getResult()
             : b.create<POP::StackAllocationOp>(
                   /*markedLifetimes=*/true,
                   PointerType::get(loweredClosureType));
}

LogicalResult ClosureLifter::liftClosureInit(ClosureInitOp closureInit,
                                             GeneratorOp generator) {
  ImplicitLocOpBuilder b(closureInit->getLoc(), closureInit.getContext());
  ClosureType closureType = getClosureType(closureInit);
  StringRef regionName = closureType.getName();
  ClosureMemoryKind memoryKind = closureType.getClosureMemoryKind();
  b.setInsertionPoint(closureInit);
  llvm::SetVector<Value> captures;
  Region &region = closureInit->getRegions().front();
  mlir::getUsedValuesDefinedAbove(region, captures);
  DenseMap<Value, Attribute> captureToSymbol;
  for (auto [capture, symbol] :
       llvm::zip(closureInit.getCaptures(),
                 closureInit.getMoveOrCopyCaptureSymbols()))
    captureToSymbol[capture] = symbol;

  // Enforce that all captures specify a capture convention.
  SmallVector<Type> fieldTypes;
  SmallVector<Capture> captureMechanisms;
  bool violatedCapturePolicy = false;
  for (Value capture : captures) {
    auto ptr = captureToSymbol.find(capture);
    if (ptr == captureToSymbol.end()) {
      violatedCapturePolicy = true;
      mlir::emitError(capture.getLoc())
          << "value is a capture of closure " << regionName
          << " but is not in the capture list";
      continue;
    }
  }

  // If there are capture without capture semantic information we cannot replace
  // them which means we cannot lift the region into a function because it is
  // not isolated.
  if (violatedCapturePolicy)
    return failure();

  // Create the capture struct type.
  for (Value capture : closureInit.getCaptures()) {
    auto ptr = captureToSymbol.find(capture);
    assert(ptr != captureToSymbol.end() && "capture must be in capture list");
    if (auto symbol = dyn_cast<SymbolConstantAttr>(ptr->second)) {
      Type capturingType =
          cast<PointerType>(capture.getType()).getElementType();
      fieldTypes.push_back(capturingType);
      captureMechanisms.push_back({symbol, capture});
      continue;
    }
    fieldTypes.push_back(capture.getType());
    captureMechanisms.push_back({{}, capture});
  }
  bool isThin = fieldTypes.empty();
  ClosureInitData closureInitData(closureType, closureInit, generator);
  Value replacement;
  if (isThin) {
    replacement = liftThinClosure(b, closureInitData,
                                  /*isRegisterPassable=*/memoryKind ==
                                      ClosureMemoryKind::REGISTER_PASSABLE);
  } else {
    Type loweredClosureType = StructType::get(fieldTypes);
    switch (memoryKind) {
    case ClosureMemoryKind::REGISTER_PASSABLE:
      replacement = liftRegPassableClosure(
          b, closureInitData, captureMechanisms, loweredClosureType);
      break;
    case ClosureMemoryKind::ESCAPING:
    case ClosureMemoryKind::NONESCAPING:
      replacement = liftNonRegPassableClosure(
          b, closureInitData, captureMechanisms, loweredClosureType);
      break;
    }
  }

  closureInit.getResult().replaceAllUsesWith(replacement);
  closureInit.erase();
  return success();
}

// lift closures and replace closure.init
void OutlineClosuresNewPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  ClosureLifter lifter(symtab);
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    bool hasFailure = false;
    generator.walk([&](ClosureInitOp closureInit) {
      hasFailure =
          hasFailure | failed(lifter.liftClosureInit(closureInit, generator));
    });
    if (hasFailure)
      return signalPassFailure();

    // update all references to the closure with the lifted symbols and struct
    // types.
    mlir::AttrTypeReplacer replacer;
    hasFailure = false;
    replacer.addReplacement([&](ClosureSymbolAttr attr) -> Attribute {
      auto ptr = lifter.liftedClosureSymbols.find(attr);
      if (ptr != lifter.liftedClosureSymbols.end())
        return ptr->second;
      mlir::emitError(theModule.getLoc())
          << "no lifted closure method found for closure symbol " << attr;
      hasFailure = true;
      return attr;
    });
    replacer.addReplacement([&](ClosureType type) -> Type {
      auto ptr = lifter.closureTypeToStructTypes.find(type);
      if (ptr != lifter.closureTypeToStructTypes.end())
        return ptr->second;
      mlir::emitError(theModule.getLoc())
          << "no lifted closure struct found for closure type " << type;
      hasFailure = true;
      return type;
    });
    generator.walk([&](Operation *op) {
      replacer.recursivelyReplaceElementsIn(op, true, true, true);
    });
    if (hasFailure)
      return signalPassFailure();
  }
}
