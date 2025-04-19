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
  ClosureLifter(SymbolTable &symtab, ParameterCollector::Analysis &paramCache)
      : counter(0), symtab(symtab), paramCache(paramCache) {}
  /// Given components of the lifted function, generate a closure symbol, which
  /// is an abstraction of a symbol used to reference functions that do not yet
  /// exist.
  ClosureSymbolAttr createClosureSymbolAttr(GeneratorOp parent, StringRef name,
                                            ClosureMethod method,
                                            ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resultTypes,
                                            ClosureType closureType,
                                            ArrayRef<Type> params,
                                            ParamClosureType closureParamType);
  /// Given a closure init op, generate functions for call, copy, move, and del
  /// + struct instance to store captures.
  LogicalResult liftClosureInit(ClosureInitOp closureInit,
                                GeneratorOp generator);

  /// Symbol name uniquer requires a counter.
  unsigned counter;
  /// The symbol table of the module.
  SymbolTable &symtab;

  /// The paramCache is an interpass cache that optimizes attribute/type walks
  /// by halting traversals at attributes/types known to not contain any
  /// parameters.
  ParameterCollector::Analysis &paramCache;

  /// Pair a parameter value with the closure attr so that we can replace the
  /// abstraction with the calculated type. For example, we may calculate that
  /// the closure "fn" inside parent @foo captured three parameters A,B,C in
  /// which case we'd like to replace the attribute #kgen.closure<@foo "fn">
  /// with the struct attr <{A,B,C}>.
  DenseMap<ClosureAttr, TypedAttr> paramCaptureToStructAttr;

  /// Pair a closure parameter type with the type of the generated parameter
  /// capture. The parameter capture is either none type in the case of no
  /// captures, the type of the captured parameter in the case of a single
  /// capture, or a struct type in the case of multiple captures.
  DenseMap<ParamClosureType, Type> paramClosureTypeToType;

  /// Pair a closure symbol with the symbol of the lifted function so that the
  /// closure symbols can be replaced.
  DenseMap<ClosureSymbolAttr, SymbolConstantAttr> liftedClosureSymbols;
  /// Pair the closure type with the struct type of the generated capture struct
  /// so that the closure types can be replaced.
  DenseMap<ClosureType, Type> closureTypeToStructTypes;
  struct ClosureInitData {
    ClosureInitData(llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
                    ClosureType closureType, ClosureInitOp closureInit,
                    GeneratorOp generator,
                    SmallVector<SymbolConstantAttr> &&moveSymbols);
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
    ArrayRef<ParamDeclAttr> getCapturedParamDecls() const {
      return ArrayRef(capturedParamDecls.begin(), capturedParamDecls.end());
    }
    ArrayRef<SymbolConstantAttr> getMoveSymbols() const {
      return ArrayRef(moveSymbols.begin(), moveSymbols.end());
    }
    ParamDeclAttr getSelfParam() const { return selfParam; }
    ParamClosureType getParamClosureType() const { return paramClosureType; }

  private:
    llvm::SetVector<ParamDeclAttr> capturedParamDecls;
    ClosureType closureType;
    ClosureInitOp closureInit;
    GeneratorOp generator;
    SmallVector<SymbolConstantAttr> moveSymbols;
    ParamDeclAttr selfParam;
    ParamClosureType paramClosureType;
  };

private:
  /// Lift a register passable closure. The characterization is in the lifted
  /// signature: a register passable closure's lifted call function has an
  /// implicit self argument of struct type.
  Value liftRegPassableClosure(ImplicitLocOpBuilder &b, ClosureInitData &data,
                               TypedAttr capturedInstance,
                               ArrayRef<Capture> captureMechanisms,
                               Type loweredClosureType);
  /// Lift a closure with no captures. We can skip the loading/storing of
  /// captures and the self type is none or opaque pointer, depending on the
  /// register passable flag.
  Value liftThinClosure(ImplicitLocOpBuilder &b, ClosureInitData &data,
                        TypedAttr capturedInstance, bool isRegisterPassable);
  /// Lift a non-register passable closure. The characterization is in the
  /// lifted signature: a non-register passable closure's lifted call function
  /// has an implicit self argument of pointer type.
  Value liftNonRegPassableClosure(ImplicitLocOpBuilder &b,
                                  ClosureInitData &closureInitData,
                                  TypedAttr capturedInstance,
                                  ArrayRef<Capture> captureMechanisms,
                                  Type loweredClosureType);
  /// Given closure metadata, lift the region of the closure init into a top
  /// level function.
  void liftCallFunction(ImplicitLocOpBuilder &b, ClosureInitData &data,
                        TypedAttr capturedInstance);
  void liftMoveFunction(ImplicitLocOpBuilder &b, ClosureInitData &data,
                        Type loweredClosureType,
                        ArrayRef<Capture> captureMechanisms,
                        TypedAttr capturedInstance);
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
                    TypedAttr capturedInstance,
                    ArrayRef<Capture> captureMechanisms,
                    Type loweredClosureType,
                    function_ref<Value(Capture, int, Value)> replacementFn);

  llvm::SetVector<ParamDeclAttr>
  collectCapturedParams(llvm::SetVector<Value> const &captures,
                        GeneratorOp generator, Region &region);
};
} // namespace

ClosureLifter::ClosureInitData::ClosureInitData(
    llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
    ClosureType closureType, ClosureInitOp closureInit, GeneratorOp generator,
    SmallVector<SymbolConstantAttr> &&moveSymbols)
    : capturedParamDecls(std::move(capturedParamDecls)),
      closureType(closureType), closureInit(closureInit), generator(generator),
      moveSymbols(std::move(moveSymbols)) {
  // Create the capture struct.
  SmallVector<Type> paramTypes;
  MLIRContext *cxt = generator->getContext();
  for (ParamDeclAttr paramCaptures : getCapturedParamDecls())
    paramTypes.push_back(paramCaptures.getType());
  Type closureParamCapture;
  StringAttr captureName;
  switch (paramTypes.size()) {
  case 0:
    closureParamCapture = KGEN::NoneType::get(cxt);
    captureName = StringAttr::get(cxt, "CAPTURES");
    break;
  case 1:
    closureParamCapture = paramTypes.front();
    captureName = getCapturedParamDecls().front().getName();
    break;
  default:
    captureName = StringAttr::get(cxt, "CAPTURES");
    closureParamCapture = StructType::get(paramTypes);
  }

  selfParam = ParamDeclAttr::get(captureName, closureParamCapture);
  paramClosureType =
      ParamClosureType::get(cxt, SymbolRefAttr::get(generator.getSymNameAttr()),
                            StringAttr::get(cxt, regionName()));
}

/// Given a region of a function assumed to have a parameter of the closure self
/// param type, unpack the parameter. For example, suppose we had a closure that
/// captured parameters C and D. We lifted the function from its nested location
/// into a top level function. Now these references are detached from the scope
/// where the declarations live. We want to reattach them to the self parameter.
/// To do so, we unpack the self param like so:
///
/// kgen.generator @lifted_closure<SELF: struct<(index, index)>>(%arg0:
/// !kgen.pointer<struct<(index)>>) -> index {
///
///   kgen.param.declare C = <#kgen.struct.extract<:struct<(index, index)> SELF,
///   0>>
///  kgen.param.declare D = <#kgen.struct.extract<:struct<(index,index)>
///   SELF, 1>>
///
///   ... now the original references to "C" and "D" are referencing parameters
///   in this scope rather than the parent scope it was lifted from.
static void
unpackCapturesInto(ImplicitLocOpBuilder &b, Region &region,
                   ClosureLifter::ClosureInitData &closureInitData) {
  // Only structs need unpacking.
  if (closureInitData.getCapturedParamDecls().size() <= 1)
    return;
  ParamDeclRefAttr selfParamRef =
      ParamDeclRefAttr::get(closureInitData.getSelfParam());
  b.setInsertionPointToStart(&region.front());
  SmallVector<TypedAttr> values;
  for (auto [index, paramCapture] :
       llvm::enumerate(closureInitData.getCapturedParamDecls())) {
    TypedAttr extractedMember = StructExtractAttr::get(
        b.getContext(), selfParamRef, index, paramCapture.getType());
    b.create<ParamDeclareOp>(
        ParamDeclAttr::get(paramCapture.getName(), paramCapture.getType()),
        extractedMember);
    values.push_back(ParamDeclRefAttr::get(paramCapture));
  }
}

ClosureSymbolAttr ClosureLifter::createClosureSymbolAttr(
    GeneratorOp parent, StringRef name, ClosureMethod method,
    ArrayRef<Type> argTypes, ArrayRef<Type> resultTypes,
    ClosureType closureType, ArrayRef<Type> params,
    ParamClosureType closureParamType) {
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
      M::KGEN::FuncTypeGeneratorType::get(params, originalFuncType);
  SmallVector<TypedAttr> boundParams;
  llvm::append_range(boundParams,
                     llvm::map_to_vector(params, [&](Type type) -> TypedAttr {
                       return UnboundAttr::get(parent.getContext(), type);
                     }));
  boundParams.push_back(ClosureAttr::get(cxt, closureParamType));
  return ClosureSymbolAttr::get(
      cxt, SymbolRefAttr::get(parent.getSymNameAttr()),
      StringAttr::get(cxt, name), ClosureMethodAttr::get(cxt, method),
      boundParams, originalfuncGenType);
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

void ClosureLifter::liftMoveFunction(ImplicitLocOpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     Type loweredClosureType,
                                     ArrayRef<Capture> captureMechanisms,
                                     TypedAttr capturedInstance) {
  // create signature
  Type selfType = closureInitData.selfType(loweredClosureType);
  GeneratorOp generator = closureInitData.getGenerator();
  SmallVector<Type> argTypes;
  argTypes.push_back(selfType);
  argTypes.push_back(selfType);
  FunctionType funcType = FunctionType::get(b.getContext(), argTypes, {});
  FuncTypeGeneratorType funcGenType =
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          closureInitData.getSelfParam(), funcType, /*argConv=*/{},
          /*effects=*/{},
          /*fnMetadata=*/{}, /*genMetadata=*/{});

  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + "_move_" + closureInitData.regionName()).str(),
      symtab, counter));
  b.setInsertionPoint(generator);
  auto moveGenerator = b.create<GeneratorOp>(uniqueName, funcGenType, funcType,
                                             closureInitData.getSelfParam());
  symtab.insert(moveGenerator);

  // Populate move body.
  Block &moveBlock = moveGenerator.getBodyRegion().emplaceBlock();
  for (Type type : argTypes)
    moveBlock.addArgument(type, moveGenerator.getLoc());
  b.setInsertionPointToStart(&moveBlock);
  Value source = moveBlock.getArgument(0);
  Value target = moveBlock.getArgument(1);
  unsigned moveIndex = 0;
  for (auto [index, capture] : llvm::enumerate(captureMechanisms)) {
    Value targetField = b.create<KGEN::StructGEPOp>(target, index);
    Value sourceField = b.create<KGEN::StructGEPOp>(source, index);
    if (!capture.moveOrCopySym.has_value()) {
      b.create<POP::StoreOp>(b.create<POP::LoadOp>(sourceField), targetField);
    } else {
      SymbolConstantAttr moveSymbol =
          closureInitData.getMoveSymbols()[moveIndex++];
      b.create<KGEN::CallOp>(moveSymbol, ValueRange{sourceField, targetField});
    }
  }
  b.create<KGEN::ReturnOp>(ValueRange{});
  unpackCapturesInto(b, moveGenerator.getBodyRegion(), closureInitData);

  // Map from synthesized function to abstracted symbols.
  SmallVector<TypedAttr> boundParams;
  boundParams.push_back(capturedInstance);
  auto sym = SymbolConstantAttr::get(
      moveGenerator,
      FuncTypeGeneratorType::get({}, funcType, /*argConv=*/{},
                                 /*effects=*/{},
                                 /*fnMetadata=*/{}, /*genMetadata=*/{}),
      boundParams);
  ClosureSymbolAttr closureAttr = createClosureSymbolAttr(
      closureInitData.getGenerator(), closureInitData.regionName(),
      ClosureMethod::MOVE, {PointerType::get(closureInitData.getClosureType())},
      {}, closureInitData.getClosureType(), {},
      closureInitData.getParamClosureType());
  liftedClosureSymbols[closureAttr] = sym;
}

void ClosureLifter::liftCallFunction(ImplicitLocOpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     TypedAttr capturedInstance) {
  Region &region = closureInitData.region();
  GeneratorOp generator = closureInitData.getGenerator();
  SmallVector<Type> argTypes;
  llvm::append_range(argTypes, region.getArgumentTypes());
  b.setInsertionPoint(generator);
  FunctionType funcType =
      FunctionType::get(b.getContext(), argTypes, closureInitData.results());
  SmallVector<ParamDeclAttr> allParams;
  append_range(allParams, closureInitData.getClosureInit().getInputParams());
  allParams.push_back(closureInitData.getSelfParam());
  FuncTypeGeneratorType funcGenType =
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          allParams, funcType, /*argConv=*/{}, /*effects=*/{},
          /*fnMetadata=*/{}, /*genMetadata=*/{});
  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + "_" + closureInitData.regionName()).str(), symtab,
      counter));
  auto liftedWrapper =
      b.create<GeneratorOp>(uniqueName, funcGenType, funcType, allParams);
  symtab.insert(liftedWrapper);

  // Remap the symbol to not include the self param by only using input params
  // and binding the final parameter.
  SmallVector<TypedAttr> boundParams = llvm::map_to_vector(
      closureInitData.getClosureInit().getInputParams(),
      [&](ParamDeclAttr attr) -> TypedAttr {
        return UnboundAttr::get(b.getContext(), attr.getType());
      });
  boundParams.push_back(capturedInstance);
  auto sym = SymbolConstantAttr::get(
      liftedWrapper,
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          closureInitData.getClosureInit().getInputParams(), funcType,
          /*argConv=*/{}, /*effects=*/{},
          /*fnMetadata=*/{}, /*genMetadata=*/{}),
      boundParams);
  Region &body = liftedWrapper.getBodyRegion();
  body.takeBody(region);
  unpackCapturesInto(b, body, closureInitData);

  // The closure symbol does not have the implicit argument; remove it
  argTypes.erase(argTypes.begin());
  SmallVector<Type> paramsUnmapped;
  for (ParamDeclAttr param : closureInitData.getClosureInit().getInputParams())
    paramsUnmapped.push_back(param.getType());
  ClosureSymbolAttr closureAttr = createClosureSymbolAttr(
      closureInitData.getGenerator(), closureInitData.regionName(),
      ClosureMethod::CALL, argTypes, closureInitData.results(),
      closureInitData.getClosureType(), paramsUnmapped,
      closureInitData.getParamClosureType());
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
    TypedAttr capturedInstance, ArrayRef<Capture> captureMechanisms,
    Type loweredClosureType,
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
  // Synthesize methods.
  liftCallFunction(b, closureInitData, capturedInstance);

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
                                            TypedAttr capturedInstance,
                                            ArrayRef<Capture> captureMechanisms,
                                            Type loweredClosureType) {
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    return b.create<KGEN::StructExtractOp>(captureStructArg, index)
        ->getResults()
        .front();
  };
  Value captureStruct =
      liftClosure(b, closureInitData, capturedInstance, captureMechanisms,
                  loweredClosureType, replacementFn);
  return b.create<POP::LoadOp>(captureStruct);
}

Value ClosureLifter::liftNonRegPassableClosure(
    ImplicitLocOpBuilder &b, ClosureInitData &closureInitData,
    TypedAttr capturedInstance, ArrayRef<Capture> captureMechanisms,
    Type loweredClosureType) {
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    Value replacement = b.create<KGEN::StructGEPOp>(captureStructArg, index);
    if (!capture.moveOrCopySym.has_value())
      replacement = b.create<POP::LoadOp>(replacement);
    return replacement;
  };
  Value captureStruct =
      liftClosure(b, closureInitData, capturedInstance, captureMechanisms,
                  loweredClosureType, replacementFn);
  liftMoveFunction(b, closureInitData, loweredClosureType, captureMechanisms,
                   capturedInstance);
  return captureStruct;
}

Value ClosureLifter::liftThinClosure(ImplicitLocOpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     TypedAttr capturedInstance,
                                     bool isRegisterPassable) {
  Type loweredClosureType = KGEN::NoneType::get(b.getContext());
  Type selfType = isRegisterPassable ? loweredClosureType
                                     : PointerType::get(loweredClosureType);
  Region &region = closureInitData.region();
  region.insertArgument((unsigned)0, selfType, region.getLoc());
  liftCallFunction(b, closureInitData, capturedInstance);
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

llvm::SetVector<ParamDeclAttr>
ClosureLifter::collectCapturedParams(llvm::SetVector<Value> const &captures,
                                     GeneratorOp generator, Region &region) {
  llvm::SetVector<ParamDeclAttr> capturedParamDecls;
  ParameterUseDefGraph uses(generator.getBodyRegion());
  uses.calculate(paramCache);

  auto regionalUseDefGraph = uses.nestedScopes.find(&region);
  assert(regionalUseDefGraph != uses.nestedScopes.end());

  // Scan the captured values for captured parameters.
  ParameterCollector collector(paramCache);
  SmallVector<ParamDeclRefAttr, 16> capturedUses;
  for (Value capture : captures) {
    bool unused = false;
    {
      VerboseCompilerTimeTraceScope traceScope("collectParameters");
      collector.collectUsesFromType(capture.getType(), capturedUses, unused);
    }
  }

  // TODO (MOCO-1660): Scan locations for captured parameters when in a debug
  // build.

  // Add all parameter uses that were defined above to the capture set.
  for (ParamDeclRefAttr paramCapture :
       regionalUseDefGraph->second.usesFromAbove) {
    auto decl =
        ParamDeclAttr::get(paramCapture.getName(), paramCapture.getType());
    capturedParamDecls.insert(decl);
  }
  return capturedParamDecls;
}

static TypedAttr
createCaptureAttribute(ImplicitLocOpBuilder &b,
                       ClosureLifter::ClosureInitData &closureInitData) {
  TypedAttr capturedInstance;
  switch (closureInitData.getCapturedParamDecls().size()) {
  case 0:
    capturedInstance = NoneAttr::get(b.getContext());
    break;
  case 1: {
    ParamDeclAttr paramCapture =
        closureInitData.getCapturedParamDecls().front();
    capturedInstance = ParamDeclRefAttr::get(paramCapture);
    break;
  }
  default: {
    b.setInsertionPoint(closureInitData.getClosureInit());
    StructAttr captureInstance = StructAttr::get(
        llvm::map_to_vector(closureInitData.getCapturedParamDecls(),
                            [](ParamDeclAttr attr) -> TypedAttr {
                              return ParamDeclRefAttr::get(attr);
                            }),
        cast<StructType>(closureInitData.getSelfParam().getType()));
    ParamDeclAttr paramDeclAttr = ParamDeclAttr::get(
        b.getStringAttr("CAPTURES"), closureInitData.getSelfParam().getType());
    b.create<ParamDeclareOp>(paramDeclAttr, captureInstance);
    capturedInstance = ParamDeclRefAttr::get(paramDeclAttr);
    break;
  }
  }
  return capturedInstance;
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

  llvm::SetVector<ParamDeclAttr> capturedParamDecls =
      collectCapturedParams(captures, generator, region);

  // Create the capture struct type and collect symbols.
  // In order to create the move constructor, we need the move constructors of
  // all capture by copy/move values.
  SmallVector<SymbolConstantAttr> moveSymbols;
  for (Value capture : closureInit.getCaptures()) {
    auto ptr = captureToSymbol.find(capture);
    assert(ptr != captureToSymbol.end() && "capture must be in capture list");
    if (auto triple = dyn_cast<MemSymbolTripleAttr>(ptr->second)) {
      SymbolConstantAttr symbol = cast<SymbolConstantAttr>(
          triple.getCopy() ? triple.getCopy() : triple.getMove());
      moveSymbols.push_back(cast<SymbolConstantAttr>(triple.getMove()));
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
  ClosureInitData closureInitData(std::move(capturedParamDecls), closureType,
                                  closureInit, generator,
                                  std::move(moveSymbols));
  // Replace parameter abstractions.
  TypedAttr capturedInstance = createCaptureAttribute(b, closureInitData);
  ClosureAttr captureAttr =
      ClosureAttr::get(b.getContext(), closureInitData.getParamClosureType());
  paramCaptureToStructAttr[captureAttr] = capturedInstance;
  paramClosureTypeToType[closureInitData.getParamClosureType()] =
      closureInitData.getSelfParam().getType();

  // Replace runtime abstractions.
  Value replacement;
  if (isThin) {
    replacement = liftThinClosure(b, closureInitData, capturedInstance,
                                  /*isRegisterPassable=*/memoryKind ==
                                      ClosureMemoryKind::REGISTER_PASSABLE);
  } else {
    Type loweredClosureType = StructType::get(fieldTypes);
    switch (memoryKind) {
    case ClosureMemoryKind::REGISTER_PASSABLE:
      replacement =
          liftRegPassableClosure(b, closureInitData, capturedInstance,
                                 captureMechanisms, loweredClosureType);
      break;
    case ClosureMemoryKind::ESCAPING:
    case ClosureMemoryKind::NONESCAPING:
      replacement =
          liftNonRegPassableClosure(b, closureInitData, capturedInstance,
                                    captureMechanisms, loweredClosureType);
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
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();
  ClosureLifter lifter(symtab, paramCache);
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
    replacer.addReplacement([&](ClosureAttr attr) -> Attribute {
      auto ptr = lifter.paramCaptureToStructAttr.find(attr);
      if (ptr != lifter.paramCaptureToStructAttr.end())
        return ptr->second;
      mlir::emitError(theModule.getLoc())
          << "no capture struct attr found for closure attr " << attr;
      return attr;
    });
    replacer.addReplacement([&](ParamClosureType type) -> Type {
      auto ptr = lifter.paramClosureTypeToType.find(type);
      if (ptr != lifter.paramClosureTypeToType.end())
        return ptr->second;
      mlir::emitError(theModule.getLoc())
          << "no type found for paramclosure type " << type;
      return type;
    });
    generator.walk([&](Operation *op) {
      replacer.recursivelyReplaceElementsIn(op, true, true, true);
    });
    if (hasFailure)
      return signalPassFailure();
  }
}
