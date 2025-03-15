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
};
} // namespace

ClosureSymbolAttr ClosureLifter::createClosureSymbolAttr(
    GeneratorOp parent, StringRef name, ClosureMethod method,
    ArrayRef<Type> argTypes, ArrayRef<Type> resultTypes,
    ClosureType closureType) {
  MLIRContext *cxt = parent->getContext();
  SmallVector<Type> originalArgTypes;
  Type selfType = PointerType::get(closureType);
  originalArgTypes.push_back(selfType);
  llvm::append_range(originalArgTypes, argTypes);
  FunctionType originalFuncType =
      FunctionType::get(cxt, originalArgTypes, resultTypes);
  M::KGEN::FuncTypeGeneratorType originalfuncGenType =
      M::KGEN::FuncTypeGeneratorType::get({}, originalFuncType);
  auto closureAttr = ClosureSymbolAttr::get(
      cxt, SymbolRefAttr::get(parent.getSymNameAttr()),
      StringAttr::get(cxt, name), ClosureMethodAttr::get(cxt, method),
      originalfuncGenType);
  return closureAttr;
}

LogicalResult ClosureLifter::liftClosureInit(ClosureInitOp closureInit,
                                             GeneratorOp generator) {
  ImplicitLocOpBuilder b(closureInit->getLoc(), closureInit.getContext());
  Type closureTypeMaybe =
      dyn_cast<PointerType>(closureInit.getResult().getType()).getElementType();
  assert(closureTypeMaybe && "closure type must be a pointer");
  ClosureType closureType = dyn_cast<ClosureType>(closureTypeMaybe);
  StringRef regionName = closureType.getName();
  b.setInsertionPoint(closureInit);

  assert(closureType && "closure init must be of closure type");
  llvm::SetVector<Value> captures;
  Region &region = closureInit->getRegions().front();
  mlir::getUsedValuesDefinedAbove(region, captures);
  DenseMap<Value, Attribute> captureToSymbol;
  for (auto [capture, symbol] :
       llvm::zip(closureInit.getCaptures(),
                 closureInit.getMoveOrCopyCaptureSymbols()))
    captureToSymbol[capture] = symbol;

  // Create the closure struct type.
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
  // If there are capture without capture semantic information we cannot replace
  // them which means we cannot lift the region into a function because it is
  // not isolated.
  if (violatedCapturePolicy)
    return failure();

  // Remove captures by augmenting region with closure struct.
  bool needsCaptureStruct = !fieldTypes.empty();
  Type loweredClosureType;
  PointerType ptrType;
  if (needsCaptureStruct) {
    loweredClosureType = StructType::get(fieldTypes);
    ptrType = PointerType::get(loweredClosureType);
    Value captureStructArg =
        region.insertArgument((unsigned)0, ptrType, closureInit.getLoc());
    b.setInsertionPointToStart(&region.front());
    for (auto [index, capture] : llvm::enumerate(captureMechanisms)) {
      Value slot = b.create<KGEN::StructGEPOp>(captureStructArg, index);
      if (capture.moveOrCopySym.has_value()) {
        replaceAllUsesInRegionWith(capture.origin, slot, region);
      } else {
        Value field = b.create<POP::LoadOp>(slot);
        replaceAllUsesInRegionWith(capture.origin, field, region);
      }
    }
  } else {
    loweredClosureType = KGEN::NoneType::get(b.getContext());
    ptrType = PointerType::get(loweredClosureType);
    region.insertArgument((unsigned)0, ptrType, closureInit.getLoc());
  }

  // Lift the region into a top level function.
  b.setInsertionPoint(generator);
  FunctionType funcType =
      FunctionType::get(b.getContext(), region.getArgumentTypes(),
                        closureInit.getFunctionType().getResults());
  M::KGEN::FuncTypeGeneratorType funcGenType =
      M::KGEN::FuncTypeGeneratorType::get({}, funcType);
  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + "_" + regionName).str(), symtab, counter));
  auto liftedWrapper = b.create<GeneratorOp>(uniqueName, funcGenType);
  symtab.insert(liftedWrapper);
  auto sym = SymbolConstantAttr::get(liftedWrapper);
  Region &body = liftedWrapper.getBodyRegion();
  body.takeBody(region);

  // create the closure attribute for the call method.
  // TODO: just use a single value in the ClosureSymbol instead of three
  // attributes
  ClosureSymbolAttr closureAttr = createClosureSymbolAttr(
      generator, regionName, ClosureMethod::CALL,
      closureInit.getFunctionType().getInputs(),
      closureInit.getFunctionType().getResults(), closureType);
  liftedClosureSymbols[closureAttr] = sym;

  // Store the captures into the capture struct.
  b.setInsertionPoint(closureInit);
  Value captureStruct;
  if (needsCaptureStruct) {
    switch (closureType.getClosureMemoryKind()) {
    case ClosureMemoryKind::ESCAPING:
      captureStruct = allocateHeapMemory(ptrType, b);
      break;
    case ClosureMemoryKind::NONESCAPING:
      captureStruct =
          b.create<POP::StackAllocationOp>(/*markedLifetimes=*/true, ptrType);
      break;
    case ClosureMemoryKind::REGISTER_PASSABLE: {
      mlir::emitError(closureInit.getLoc(),
                      "register passable closures not yet implemented");
      return failure();
    }
    }
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
  } else {
    // TODO: MOCO-1702 (optimize these away)
    // This is a dead argument.
    captureStruct =
        b.create<POP::StackAllocationOp>(/*markedLifetimes=*/true, ptrType);
  }

  closureInit.getResult().replaceAllUsesWith(captureStruct);
  closureInit.erase();
  closureTypeToStructTypes[closureType] = loweredClosureType;
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
