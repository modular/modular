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
#include "mlir/Transforms/RegionUtils.h"

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
  /// the symbol of the destructor, if applicable
  std::optional<SymbolConstantAttr> delSym;

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
  Value sizeOf = b.create<ParamConstantOp>(
      loc, ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = b.create<ParamConstantOp>(
      loc, ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return b.create<POP::AlignedAllocOp>(loc, ptrType,
                                       ValueRange{alignOf, sizeOf});
}

namespace {
/// The ClosureLifter is responsible for
/// (a) lifting a closure init into a top level function + capture struct and
/// (b) storing metadata necessary to replace references to the closure.
struct ClosureLifter {
  ClosureLifter(SymbolTable &symtab, ParameterCollector::Analysis &paramCache,
                bool debugBuild)
      : counter(0), symtab(symtab), paramCache(paramCache),
        debugBuild(debugBuild) {}
  /// Given a closure init op, generate functions for call, copy, move, and del
  /// + struct instance to store captures.
  LogicalResult liftClosureInit(ClosureInitOp closureInit,
                                GeneratorOp generator,
                                StructGeneratorOp generatorOp);

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
  DenseMap<ClosureType, Type> packedClosureType;

  /// True if built with debug metadata.
  bool debugBuild;
  struct ClosureInitData {
    ClosureInitData(
        llvm::SetVector<ParamDeclAttr> const &&capturedParamDecls,
        ClosureType closureType, ClosureInitOp closureInit,
        StructGeneratorOp structGeneratorOp, GeneratorOp generator,
        SmallVector<SymbolConstantAttr> &&moveSymbols,
        std::optional<SmallVector<SymbolConstantAttr>> &&copySymbols);
    Type selfType(Type loweredClosureType) const {
      return closureType.getClosureMemoryKind() == ClosureMemoryKind::TRIVIAL
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
    StringRef getCapturesParamName() const { return capturedParametersName; }
    bool isEscaping() const {
      return closureType.getClosureMemoryKind() == ClosureMemoryKind::ESCAPING;
    }
    bool isMem() const {
      return closureType.getClosureMemoryKind() != ClosureMemoryKind::TRIVIAL;
    }
    ArrayRef<ParamDeclAttr> getCapturedParamDecls() const {
      return ArrayRef(capturedParamDecls.begin(), capturedParamDecls.end());
    }
    ArrayRef<SymbolConstantAttr> getMoveSymbols() const {
      return ArrayRef(moveSymbols.begin(), moveSymbols.end());
    }
    ArrayRef<SymbolConstantAttr> getCopySymbols() const {
      return ArrayRef(copySymbolsMaybe->begin(), copySymbolsMaybe->end());
    }
    bool isCopyable() const { return copySymbolsMaybe.has_value(); }
    ParamDeclAttr getSelfParam() const { return selfParam; }
    ParamClosureType getParamClosureType() const { return paramClosureType; }
    ClosureSymbolAttr closureSymbolForSourceName(StringRef sourceName) const;
    Location getLiftedLocation() const { return liftedLocation; }

  private:
    llvm::SetVector<ParamDeclAttr> capturedParamDecls;
    /// The map of symbols to replace.
    DenseMap<StringRef, ClosureSymbolAttr> abstractSymbolMap;
    /// The name of the captures parameter in the corresponding struct generator
    StringRef capturedParametersName;
    ClosureType closureType;
    ClosureInitOp closureInit;
    GeneratorOp generator;
    SmallVector<SymbolConstantAttr> moveSymbols;
    std::optional<SmallVector<SymbolConstantAttr>> copySymbolsMaybe;
    ParamDeclAttr selfParam;
    ParamClosureType paramClosureType;
    Location liftedLocation;
  };

private:
  /// Lift a register passable closure. The characterization is in the lifted
  /// signature: a register passable closure's lifted call function has an
  /// implicit self argument of struct type.
  Value liftRegPassableClosure(OpBuilder &b, ClosureInitData &data,
                               TypedAttr capturedInstance,
                               ArrayRef<Capture> captureMechanisms,
                               Type loweredClosureType);
  /// Lift a closure with no captures. We can skip the loading/storing of
  /// captures and the self type is none or opaque pointer, depending on the
  /// register passable flag.
  Value liftThinClosure(OpBuilder &b, ClosureInitData &data,
                        TypedAttr capturedInstance, bool isRegisterPassable);
  /// Lift a non-register passable closure. The characterization is in the
  /// lifted signature: a non-register passable closure's lifted call function
  /// has an implicit self argument of pointer type.
  Value liftNonRegPassableClosure(OpBuilder &b,
                                  ClosureInitData &closureInitData,
                                  TypedAttr capturedInstance,
                                  ArrayRef<Capture> captureMechanisms,
                                  Type loweredClosureType);
  /// Given closure metadata, lift the region of the closure init into a top
  /// level function.
  LogicalResult liftCallFunction(OpBuilder &b, ClosureInitData &data,
                                 TypedAttr capturedInstance,
                                 Type loweredClosureType);
  void liftMoveOrCopyFunction(OpBuilder &b, ClosureInitData &data,
                              Type loweredClosureType,
                              ArrayRef<Capture> captureMechanisms,
                              TypedAttr capturedInstance, bool isMove);
  void liftDelFunction(OpBuilder &b, ClosureInitData &data,
                       Type loweredClosureType,
                       ArrayRef<Capture> captureMechanisms,
                       TypedAttr capturedInstance);

  void
  createClosureGenerator(OpBuilder &b, ClosureInitData &closureInitData,
                         ClosureMethod method, FunctionType funcType,
                         TypedAttr capturedInstance,
                         llvm::function_ref<void(GeneratorOp)> populateBody,
                         ArrayRef<ArgConvention> argConventions);

  /// Given closure metadata the captures, emit code that results in the storage
  /// of the captures into capture struct.
  void storeCaptures(OpBuilder &b, Value captureStructArg,
                     ClosureInitData &closureInitData,
                     ArrayRef<Capture> captureMechanisms);
  /// Lift driving function. The loweredClosureType should be a kgen.struct with
  /// the captures and the replacement function is meant to emit the IR
  /// necessary for extracting the capture value out of the closure struct
  /// instance.
  Value liftClosure(OpBuilder &b, ClosureInitData &closureInitData,
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
    ClosureType closureType, ClosureInitOp closureInit,
    StructGeneratorOp structGeneratorOp, GeneratorOp generator,
    SmallVector<SymbolConstantAttr> &&moveSymbols,
    std::optional<SmallVector<SymbolConstantAttr>> &&copySymbols)
    : capturedParamDecls(std::move(capturedParamDecls)),
      closureType(closureType), closureInit(closureInit), generator(generator),
      moveSymbols(std::move(moveSymbols)),
      copySymbolsMaybe(std::move(copySymbols)),
      liftedLocation(FusedLoc::get(
          generator->getContext(),
          Location(DebugInfo::extractSourceLoc(closureInit->getLoc())),
          closureInit.getSubprogramScope())) {
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
  paramClosureType = ParamClosureType::get(cxt, closureType.getParentSymbol(),
                                           StringAttr::get(cxt, regionName()));
  structGeneratorOp->walk([&](WitnessOp witness) {
    if (auto closureSym = dyn_cast<ClosureSymbolAttr>(witness.getValue()))
      abstractSymbolMap[witness.getName()] = closureSym;
  });

  // Captures parameter is always last.
  capturedParametersName = structGeneratorOp.getInputParams().back().getName();
}

ClosureSymbolAttr ClosureLifter::ClosureInitData::closureSymbolForSourceName(
    StringRef sourceName) const {
  auto sym = abstractSymbolMap.find(sourceName);
  if (sym == abstractSymbolMap.end())
    return {};
  return sym->getSecond();
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
/// Returns a map from the original captured parameter to the struct extraction
/// expression so that the mapping can be reused in the signature remapping.
static DenseMap<StringAttr, TypedAttr>
unpackCapturesInto(OpBuilder &b, Region &region,
                   ClosureLifter::ClosureInitData &closureInitData) {
  DenseMap<StringAttr, TypedAttr> fromRefToExtract;
  // Only structs need unpacking.
  if (closureInitData.getCapturedParamDecls().size() <= 1)
    return fromRefToExtract;
  ParamDeclRefAttr selfParamRef =
      ParamDeclRefAttr::get(closureInitData.getSelfParam());
  b.setInsertionPointToStart(&region.front());
  for (auto [index, paramCapture] :
       llvm::enumerate(closureInitData.getCapturedParamDecls())) {
    TypedAttr extractedMember = StructExtractAttr::get(
        b.getContext(), selfParamRef, index, paramCapture.getType());
    b.create<ParamDeclareOp>(
        closureInitData.getLiftedLocation(),
        ParamDeclAttr::get(paramCapture.getName(), paramCapture.getType()),
        extractedMember);
    fromRefToExtract[paramCapture.getName()] = extractedMember;
  }
  return fromRefToExtract;
}

/// Given
/// (1) FuncType: the original closure method signature
/// (2) fromRefToExtract: a map from parameter names to a struct extract
/// expression Return a new function type that replaces captured parameters with
/// struct extract expressions. The region is also provided so that rebinds can
/// be emitted to adapt to the different representations of the same type.
FunctionType
remapFuncType(OpBuilder &b, Region &region, FunctionType oldFuncType,
              DenseMap<StringAttr, TypedAttr> const &fromRefToExtract,
              Location loc) {
  // We need to replace the captured parameters in the arguments of the region
  // with the extract expressions
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr attr) -> Attribute {
    if (fromRefToExtract.count(attr.getName())) {
      return fromRefToExtract.at(attr.getName());
    }
    return attr;
  });

  // Remap argument types and add rebind adaptors.
  for (auto arg : region.getArguments()) {
    Type oldType = arg.getType();
    Type newType = replacer.replace(arg.getType());
    if (newType != oldType) {
      arg.setType(newType);
      Value newValue = b.create<KGEN::RebindOp>(loc, oldType, arg);
      arg.replaceAllUsesExcept(newValue, newValue.getDefiningOp());
    }
  }

  // Remap result types and add rebind adaptors.
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(oldFuncType.getResults(), [&](Type type) -> Type {
        return replacer.replace(type);
      });
  region.walk([&](ReturnOp op) {
    b.setInsertionPoint(op);
    for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
      Type oldType = operand.getType();
      Type newType = resultTypes[index];
      if (newType != oldType) {
        Value newValue = b.create<KGEN::RebindOp>(loc, newType, operand);
        operand.replaceAllUsesExcept(newValue, newValue.getDefiningOp());
      }
    }
  });

  return FunctionType::get(b.getContext(),
                           llvm::map_to_vector(region.getArguments(),
                                               [&](BlockArgument arg) -> Type {
                                                 return arg.getType();
                                               }),
                           resultTypes);
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

void ClosureLifter::createClosureGenerator(
    OpBuilder &b, ClosureInitData &closureInitData, ClosureMethod method,
    FunctionType funcType, TypedAttr capturedInstance,
    llvm::function_ref<void(GeneratorOp)> populateBody,
    ArrayRef<ArgConvention> argConventions) {
  StringRef baseName;
  switch (method) {
  case ClosureMethod::MOVE:
    baseName = "__moveinit__";
    break;
  case ClosureMethod::DEL:
    baseName = "__del__";
    break;
  case ClosureMethod::COPY:
    baseName = "__copyinit__";
    break;
  default:
    llvm_unreachable("Invalid closure method");
    break;
  }
  ClosureSymbolAttr closureAttr =
      closureInitData.closureSymbolForSourceName(baseName);
  /// If there is no witness for this method then there is no reference to it
  if (!closureAttr)
    return;
  GeneratorOp generator = closureInitData.getGenerator();
  FuncTypeGeneratorType funcGenType =
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          closureInitData.getSelfParam(), funcType, /*argConv=*/argConventions,
          /*effects=*/{},
          /*fnMetadata=*/{}, /*genMetadata=*/{});

  SmallVector<Type> resultTypes;
  resultTypes.push_back(funcType.getResult(0));

  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + baseName + closureInitData.regionName()).str(),
      symtab, counter));
  b.setInsertionPoint(generator);
  auto closureGenerator = b.create<GeneratorOp>(
      closureInitData.getLiftedLocation(), uniqueName, funcGenType, funcType,
      closureInitData.getSelfParam());
  symtab.insert(closureGenerator);

  populateBody(closureGenerator);

  auto fromParamToExtractExpr =
      unpackCapturesInto(b, closureGenerator.getBodyRegion(), closureInitData);
  auto newFuncType = remapFuncType(b, closureGenerator.getBodyRegion(),
                                   funcType, fromParamToExtractExpr,
                                   closureInitData.getLiftedLocation());
  closureGenerator.setFunctionType(newFuncType);

  // Map from synthesized function to abstracted symbols.
  SmallVector<TypedAttr> boundParams;
  boundParams.push_back(ParamDeclRefAttr::get(
      closureInitData.getCapturesParamName(), capturedInstance.getType()));
  auto sym = SymbolConstantAttr::get(
      closureGenerator,
      FuncTypeGeneratorType::get({}, newFuncType, /*argConv=*/argConventions,
                                 /*effects=*/{},
                                 /*fnMetadata=*/{}, /*genMetadata=*/{}),
      boundParams);
  liftedClosureSymbols[closureAttr] = sym;
}

void ClosureLifter::liftDelFunction(OpBuilder &b,
                                    ClosureInitData &closureInitData,
                                    Type loweredClosureType,
                                    ArrayRef<Capture> captureMechanisms,
                                    TypedAttr capturedInstance) {
  Location loc = closureInitData.getLiftedLocation();
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
        Value field = b.create<KGEN::StructGEPOp>(loc, source, index);
        SymbolConstantAttr delSymbol = *capture.delSym;
        b.create<KGEN::CallOp>(loc, delSymbol, field);
      }
    }
    auto noneAttr = b.create<KGEN::ParamConstantOp>(
        loc, KGEN::NoneAttr::get(b.getContext()));
    b.create<KGEN::ReturnOp>(loc, noneAttr->getResults().front());
  };
  createClosureGenerator(b, closureInitData, ClosureMethod::DEL, funcType,
                         capturedInstance, populateBody,
                         {ArgConvention::OwnedMem});
}

void ClosureLifter::liftMoveOrCopyFunction(OpBuilder &b,
                                           ClosureInitData &closureInitData,
                                           Type loweredClosureType,
                                           ArrayRef<Capture> captureMechanisms,
                                           TypedAttr capturedInstance,
                                           bool isMove) {
  Location loc = closureInitData.getLiftedLocation();
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
      Value targetField = b.create<KGEN::StructGEPOp>(loc, target, index);
      Value sourceField = b.create<KGEN::StructGEPOp>(loc, source, index);
      if (!capture.moveOrCopySym.has_value()) {
        b.create<POP::StoreOp>(loc, b.create<POP::LoadOp>(loc, sourceField),
                               targetField);
      } else {
        SymbolConstantAttr symbol;
        if (isMove)
          symbol = closureInitData.getMoveSymbols()[symIndex++];
        else
          symbol = closureInitData.getCopySymbols()[symIndex++];
        b.create<KGEN::CallOp>(loc, symbol,
                               ValueRange{sourceField, targetField});
      }
    }
    auto noneAttr = b.create<KGEN::ParamConstantOp>(
        loc, KGEN::NoneAttr::get(b.getContext()));
    b.create<KGEN::ReturnOp>(loc, noneAttr->getResults().front());
  };
  createClosureGenerator(
      b, closureInitData, isMove ? ClosureMethod::MOVE : ClosureMethod::COPY,
      funcType, capturedInstance, populateBody,
      {isMove ? ArgConvention::OwnedMem : ArgConvention::ReadMem,
       ArgConvention::ByRefResult});
}

static bool typesMatch(FuncTypeGeneratorType closureSymbolAttrType,
                       FuncTypeGeneratorType liftedFunctionType,
                       Type loweredClosureType,
                       ClosureLifter::ClosureInitData closureInitData) {

  /// (1) First map the ClosureType to the lowered struct type in the closure
  /// symbol attribute's type.
  ArrayRef<Type> args = closureSymbolAttrType.getBody().getArguments();
  MLIRContext *cxt = closureSymbolAttrType.getContext();
  /// Expected at least one self argument.
  if (args.size() < 1) {
    mlir::emitError(closureInitData.getClosureInit()->getLoc(),
                    "expected at least one argument in the struct method ")
        << closureSymbolAttrType;
    return false;
  }
  Type selfType;
  if (closureInitData.isMem()) {
    /// Expected pointer semantics for in memory closure
    if (!isa<KGEN::PointerType>(args.front())) {
      mlir::emitError(
          closureInitData.getClosureInit()->getLoc(),
          "expected a pointer type in the first argument of the method ")
          << closureSymbolAttrType;
      return false;
    }

    selfType = cast<PointerType>(args.front()).getElementType();
  } else {
    selfType = args.front();
  }
  auto closureTypeOfGiven = dyn_cast<ClosureType>(selfType);
  if (!closureTypeOfGiven) {
    mlir::emitError(
        closureInitData.getClosureInit()->getLoc(),
        "expected a closure type in the first argument of the method ")
        << closureSymbolAttrType;
    return false;
  }

  ClosureType closureType = closureInitData.getClosureType();
  if (closureTypeOfGiven.getParentSymbol() != closureType.getParentSymbol() ||
      closureTypeOfGiven.getName() != closureType.getName() ||
      closureTypeOfGiven.getClosureMemoryKind() !=
          closureType.getClosureMemoryKind()) {
    mlir::emitError(closureInitData.getClosureInit()->getLoc(),
                    "unexpected closure type. Got ")
        << closureTypeOfGiven << " but expected " << closureType;
    return false;
  }
  SmallVector<Type> loweredArgTypes;
  loweredArgTypes.push_back(closureInitData.isMem()
                                ? PointerType::get(loweredClosureType)
                                : loweredClosureType);
  llvm::append_range(loweredArgTypes, args.drop_front());
  SmallVector<ParamDeclAttr> parameters;

  FunctionType givenFuncType = FunctionType::get(
      cxt, loweredArgTypes, closureSymbolAttrType.getBody().getResults());
  // (2) Next, remap from the struct generator op parameters to the parameters
  // of the lifted function.
  M::KGEN::FuncTypeGeneratorType givenFuncGenTypeRemappedParams =
      closureSymbolAttrType.remapToFuncTypeGenerator(
          closureInitData.getClosureInit().getInputParams(), givenFuncType,
          closureSymbolAttrType.getBody().getArgConventions(),
          closureSymbolAttrType.getBody().getFnEffects());

  bool isMatch = givenFuncGenTypeRemappedParams == liftedFunctionType;
  if (!isMatch) {
    mlir::emitError(closureInitData.getClosureInit()->getLoc(),
                    "Type mismatch: ")
        << givenFuncGenTypeRemappedParams << " vs " << liftedFunctionType;
  }
  return isMatch;
}

LogicalResult ClosureLifter::liftCallFunction(OpBuilder &b,
                                              ClosureInitData &closureInitData,
                                              TypedAttr capturedInstance,
                                              Type loweredClosureType) {
  Location loc = closureInitData.getLiftedLocation();
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
  SmallVector<ArgConvention> conventions;
  ArgConvention selfConvention =
      closureInitData.isMem() ? ArgConvention::ReadMem : ArgConvention::ReadReg;
  conventions.push_back(selfConvention);
  for (auto argConvention : closureInitData.getClosureInit()
                                .getFuncTypeGenerator()
                                .getBody()
                                .getArgConventions())
    conventions.push_back(argConvention);
  FnEffects effects = closureInitData.getClosureInit()
                          .getFuncTypeGenerator()
                          .getBody()
                          .getFnEffects();
  FuncTypeGeneratorType funcGenType =
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          allParams, funcType, /*argConv=*/conventions, /*effects=*/effects,
          /*fnMetadata=*/{}, /*genMetadata=*/{});
  auto uniqueName = b.getStringAttr(getUniqueSymbolName(
      (generator.getName() + "_" + closureInitData.regionName()).str(), symtab,
      counter));
  auto liftedWrapper =
      b.create<GeneratorOp>(loc, uniqueName, funcGenType, funcType, allParams);
  liftedWrapper.setInlineLevel(
      closureInitData.getClosureInit().getInlineLevel());
  symtab.insert(liftedWrapper);

  // Remap the symbol to not include the self param by only using input params
  // and binding the final parameter.
  SmallVector<TypedAttr> boundParams = llvm::map_to_vector(
      closureInitData.getClosureInit().getInputParams(),
      [&](ParamDeclAttr attr) -> TypedAttr {
        return UnboundAttr::get(b.getContext(), attr.getType());
      });
  boundParams.push_back(ParamDeclRefAttr::get(
      closureInitData.getCapturesParamName(), capturedInstance.getType()));

  Region &body = liftedWrapper.getBodyRegion();
  body.takeBody(region);
  /// Scope verification is conditional on the enclosing function. If the
  /// function carries a fused subprogram scope, every nested op must carry a
  /// compatible scope. If the closure was pulled in from a bytecode package it
  /// will not carry a subprogram because the dibuilder in the parser is null
  /// for package building. Synthesis is required to pass verification in this
  /// case.
  DebugInfo::DIBuilder builder(b.getContext());
  DebugInfo::DISubprogramAttr scope =
      DebugInfo::extractScope((mlir::FunctionOpInterface)liftedWrapper);
  builder.pushScope(scope);
  if (failed(builder.visitLexicalRegion(liftedWrapper.getBodyRegion())))
    return failure();

  /// Get a map from a captured parameter "T" to the expression with respect to
  /// the packed parameter value, i.e. "struct.extract<CAPTURES, 0>"
  auto fromParamToExtractExpr = unpackCapturesInto(b, body, closureInitData);
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr paramDeclRefAttr) -> TypedAttr {
    auto ptr = fromParamToExtractExpr.find(paramDeclRefAttr.getName());
    if (ptr != fromParamToExtractExpr.end())
      return ptr->getSecond();
    return paramDeclRefAttr;
  });
  /// Save the closure type in terms of the packed parameter since we cannot
  /// reference undeclared parameters in the struct generator op. That is,
  /// instead of mapping the closure struct to something like
  /// `kgen.struct<(A,B)>` map the closure struct to
  /// `kgen.struct<(struct.extract<CAPTURES, 0>, struct.extract<CAPTURES, 1>)>`
  /// so that the verifier does not complain that the struct generator op did
  /// not declare A or B.
  auto loweredClosureTypeMapped = replacer.replace(loweredClosureType);
  packedClosureType[closureInitData.getClosureType()] =
      loweredClosureTypeMapped;
  FunctionType remappedFuncType =
      remapFuncType(b, body, funcType, fromParamToExtractExpr,
                    closureInitData.getLiftedLocation());
  liftedWrapper.setFunctionType(remappedFuncType);

  // The closure symbol does not have the implicit argument; remove it
  argTypes.erase(argTypes.begin());
  SmallVector<Type> paramsUnmapped;
  for (Type paramType : closureInitData.getClosureInit()
                            .getFuncTypeGenerator()
                            .getInputParamTypes())
    paramsUnmapped.push_back(paramType);
  ClosureSymbolAttr closureAttr =
      closureInitData.closureSymbolForSourceName("__call__");
  auto sym = SymbolConstantAttr::get(
      liftedWrapper,
      FuncTypeGeneratorType::remapToFuncTypeGenerator(
          closureInitData.getClosureInit().getInputParams(), remappedFuncType,
          /*argConv=*/conventions, /*effects=*/effects,
          /*fnMetadata=*/{}, /*genMetadata=*/{}),
      boundParams);
  /// We are replacing symbol A with symbol B. Ensure the types match.
  if (debugBuild) {
    if (!typesMatch(closureAttr.getType(), sym.getType(),
                    loweredClosureTypeMapped, closureInitData))
      return failure();
  }
  liftedClosureSymbols[closureAttr] = sym;
  return success();
}

void ClosureLifter::storeCaptures(OpBuilder &b, Value captureStruct,
                                  ClosureInitData &data,
                                  ArrayRef<Capture> captureMechanisms) {
  b.setInsertionPoint(data.getClosureInit());
  Location location = data.getClosureInit()->getLoc();
  for (auto [index, captureMechanism] : llvm::enumerate(captureMechanisms)) {
    auto slot = b.create<StructGEPOp>(location, captureStruct, index);
    if (captureMechanism.moveOrCopySym.has_value()) {
      SymbolConstantAttr symbol = *captureMechanism.moveOrCopySym;
      StringRef name = symbol.getSymbol().getRootReference();
      Operation *op = symtab.lookup(name);
      GeneratorOp function = cast<GeneratorOp>(op);
      SmallVector<Value> values = {captureMechanism.origin, slot};
      b.create<KGEN::CallOp>(location, function.getFunctionType().getResults(),
                             symbol, ValueRange(values));
    } else {
      b.create<POP::StoreOp>(location, captureMechanism.origin, slot);
    }
  }
}

Value ClosureLifter::liftClosure(
    OpBuilder &b, ClosureInitData &closureInitData, TypedAttr capturedInstance,
    ArrayRef<Capture> captureMechanisms, Type loweredClosureType,
    function_ref<Value(Capture, int, Value)> replacementFn) {
  Region &region = closureInitData.region();
  Location loc = closureInitData.getClosureInit().getLoc();
  // Outline Closure.
  Type selfType = closureInitData.selfType(loweredClosureType);
  Value captureStructArg = region.insertArgument(
      (unsigned)0, selfType, closureInitData.getLiftedLocation());
  b.setInsertionPointToStart(&region.front());
  for (auto [index, capture] : llvm::enumerate(captureMechanisms))
    replaceAllUsesInRegionWith(capture.origin,
                               replacementFn(capture, index, captureStructArg),
                               region);
  // Synthesize methods.
  if (failed(liftCallFunction(b, closureInitData, capturedInstance,
                              loweredClosureType)))
    return {};

  // Instantiate capture struct.
  b.setInsertionPoint(closureInitData.getClosureInit());
  Value captureStruct =
      closureInitData.isEscaping()
          ? allocateHeapMemory(cast<PointerType>(selfType), b, loc)
          : b.create<POP::StackAllocationOp>(
                 loc,
                 /*markedLifetimes=*/true, PointerType::get(loweredClosureType))
                .getResult();
  storeCaptures(b, captureStruct, closureInitData, captureMechanisms);
  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;
  return captureStruct;
}

Value ClosureLifter::liftRegPassableClosure(OpBuilder &b,
                                            ClosureInitData &closureInitData,
                                            TypedAttr capturedInstance,
                                            ArrayRef<Capture> captureMechanisms,
                                            Type loweredClosureType) {
  Location loc = closureInitData.getLiftedLocation();
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    return b.create<KGEN::StructExtractOp>(loc, captureStructArg, index)
        ->getResults()
        .front();
  };
  Value captureStruct =
      liftClosure(b, closureInitData, capturedInstance, captureMechanisms,
                  loweredClosureType, replacementFn);
  if (!captureStruct)
    return {};
  return b.create<POP::LoadOp>(closureInitData.getClosureInit()->getLoc(),
                               captureStruct);
}

Value ClosureLifter::liftNonRegPassableClosure(
    OpBuilder &b, ClosureInitData &closureInitData, TypedAttr capturedInstance,
    ArrayRef<Capture> captureMechanisms, Type loweredClosureType) {
  Location loc = closureInitData.getLiftedLocation();
  auto replacementFn = [&](Capture capture, int index, Value captureStructArg) {
    Value replacement =
        b.create<KGEN::StructGEPOp>(loc, captureStructArg, index);
    if (!capture.moveOrCopySym.has_value())
      replacement = b.create<POP::LoadOp>(loc, replacement);
    return replacement;
  };
  Value captureStruct =
      liftClosure(b, closureInitData, capturedInstance, captureMechanisms,
                  loweredClosureType, replacementFn);
  if (!captureStruct)
    return {};
  liftMoveOrCopyFunction(b, closureInitData, loweredClosureType,
                         captureMechanisms, capturedInstance, /*isMove=*/true);
  if (closureInitData.isCopyable())
    liftMoveOrCopyFunction(b, closureInitData, loweredClosureType,
                           captureMechanisms, capturedInstance,
                           /*isMove=*/false);
  liftDelFunction(b, closureInitData, loweredClosureType, captureMechanisms,
                  capturedInstance);
  return captureStruct;
}

Value ClosureLifter::liftThinClosure(OpBuilder &b,
                                     ClosureInitData &closureInitData,
                                     TypedAttr capturedInstance,
                                     bool isRegisterPassable) {
  Type loweredClosureType = KGEN::NoneType::get(b.getContext());
  Type selfType = isRegisterPassable ? loweredClosureType
                                     : PointerType::get(loweredClosureType);
  Region &region = closureInitData.region();
  region.insertArgument((unsigned)0, selfType,
                        closureInitData.getLiftedLocation());
  if (failed(liftCallFunction(b, closureInitData, capturedInstance,
                              loweredClosureType)))
    return {};
  // TODO: create thunks for register passable closures (MOCO-2242).
  if (!isRegisterPassable) {
    liftMoveOrCopyFunction(b, closureInitData, loweredClosureType, {},
                           capturedInstance, /*isMove=*/true);
    if (closureInitData.isCopyable())
      liftMoveOrCopyFunction(b, closureInitData, loweredClosureType, {},
                             capturedInstance, /*isMove=*/false);
    liftDelFunction(b, closureInitData, loweredClosureType, {},
                    capturedInstance);
  }

  closureTypeToStructTypes[closureInitData.getClosureType()] =
      loweredClosureType;
  b.setInsertionPoint(closureInitData.getClosureInit());
  Location loc = closureInitData.getClosureInit()->getLoc();
  return isRegisterPassable
             ? b.create<ParamConstantOp>(loc, NoneAttr::get(b.getContext()))
                   .getResult()
             : b.create<POP::StackAllocationOp>(
                   loc,
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
    collector.collectUsesFromType(capture.getType(), capturedUses, unused);
  }

  if (debugBuild) {
    region.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      bool unused = false;
      collector.collectUsesFromAttr(op->getLoc(), capturedUses, unused);
      return WalkResult::advance();
    });
  }

  for (auto use : capturedUses)
    capturedParamDecls.insert(ParamDeclAttr::get(use.getName(), use.getType()));

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
createCaptureAttribute(OpBuilder &b,
                       ClosureLifter::ClosureInitData &closureInitData,
                       Location loc) {
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
    b.create<ParamDeclareOp>(loc, paramDeclAttr, captureInstance);
    capturedInstance = ParamDeclRefAttr::get(paramDeclAttr);
    break;
  }
  }
  return capturedInstance;
}

LogicalResult
ClosureLifter::liftClosureInit(ClosureInitOp closureInit, GeneratorOp generator,
                               StructGeneratorOp structGeneratorOp) {
  OpBuilder b(closureInit.getContext());
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
  SmallVector<SymbolConstantAttr> copySymbols;
  bool allCopySymbolsAvailable = true;
  for (Value capture : closureInit.getCaptures()) {
    auto ptr = captureToSymbol.find(capture);
    assert(ptr != captureToSymbol.end() && "capture must be in capture list");
    if (auto triple = dyn_cast<MemSymbolTripleAttr>(ptr->second)) {
      SymbolConstantAttr symbol = cast<SymbolConstantAttr>(
          triple.getIsMove() ? triple.getMove() : triple.getCopy());
      auto moveSymbol = triple.getMove();
      auto copySymbol = triple.getCopy();
      if (moveSymbol)
        moveSymbols.push_back(cast<SymbolConstantAttr>(moveSymbol));
      else if (copySymbol)
        moveSymbols.push_back(cast<SymbolConstantAttr>(copySymbol));
      else
        llvm_unreachable("cannot capture by move or copy and not include a "
                         "move or copy symbol");
      if (copySymbol && allCopySymbolsAvailable)
        copySymbols.push_back(copySymbol);
      else
        allCopySymbolsAvailable = false;
      SymbolConstantAttr del = cast<SymbolConstantAttr>(triple.getDel());
      Type capturingType =
          cast<PointerType>(capture.getType()).getElementType();
      fieldTypes.push_back(capturingType);
      captureMechanisms.push_back({symbol, del, capture});
      continue;
    }
    fieldTypes.push_back(capture.getType());
    captureMechanisms.push_back({{}, {}, capture});
  }
  bool isThin = fieldTypes.empty();
  std::optional<SmallVector<SymbolConstantAttr>> copiesMaybe;
  if (allCopySymbolsAvailable)
    copiesMaybe = std::move(copySymbols);

  ClosureInitData closureInitData(std::move(capturedParamDecls), closureType,
                                  closureInit, structGeneratorOp, generator,
                                  std::move(moveSymbols),
                                  std::move(copiesMaybe));
  // Replace parameter abstractions.
  TypedAttr capturedInstance =
      createCaptureAttribute(b, closureInitData, closureInit->getLoc());
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
                                      ClosureMemoryKind::TRIVIAL);
  } else {

    Type loweredClosureType =
        StructType::get(b.getContext(), fieldTypes,
                        memoryKind != ClosureMemoryKind::TRIVIAL &&
                            memoryKind != ClosureMemoryKind::REGISTER_PASSABLE);
    switch (memoryKind) {
    case ClosureMemoryKind::TRIVIAL:
      replacement =
          liftRegPassableClosure(b, closureInitData, capturedInstance,
                                 captureMechanisms, loweredClosureType);
      break;
    case ClosureMemoryKind::REGISTER_PASSABLE:
    case ClosureMemoryKind::ESCAPING:
    case ClosureMemoryKind::NONESCAPING:
      replacement =
          liftNonRegPassableClosure(b, closureInitData, capturedInstance,
                                    captureMechanisms, loweredClosureType);
      break;
    }
  }
  if (!replacement)
    return failure();
  closureInit.getResult().replaceAllUsesWith(replacement);
  closureInit.erase();
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

static LogicalResult
liftClosuresFromRegion(ModuleOp theModule, SymbolTable &symtab,
                       ParameterCollector::Analysis &paramCache,
                       bool debugBuild, Operation *enclosingOp) {
  ClosureLifter lifter(symtab, paramCache, debugBuild);
  SmallVector<std::pair<ClosureType, StructGeneratorOp>> structGenerators;
  bool hasFailure = false;
  GeneratorOp parent = isa<GeneratorOp>(enclosingOp)
                           ? cast<GeneratorOp>(enclosingOp)
                           : enclosingOp->getParentOfType<GeneratorOp>();
  enclosingOp->walk([&](ClosureInitOp closureInit) {
    if (closureInit.getOperation() == enclosingOp)
      return;
    ClosureType closureType = getClosureType(closureInit);
    StringAttr symbol = getFullName(closureType);
    if (StructGeneratorOp structGeneratorOp =
            symtab.lookup<StructGeneratorOp>(symbol)) {
      hasFailure = hasFailure | failed(lifter.liftClosureInit(
                                    closureInit, parent, structGeneratorOp));
      structGenerators.push_back(std::pair<ClosureType, StructGeneratorOp>(
          closureType, structGeneratorOp));
    } else {
      mlir::emitError(theModule.getLoc())
          << "missing struct generator op for closure "
          << getClosureType(closureInit).getName();
      hasFailure = true;
    }
  });
  if (hasFailure)
    return failure();
  if (lifter.closureTypeToStructTypes.empty())
    return success();
  // update all references to the closure with the lifted symbols and struct
  // types.
  mlir::AttrTypeReplacer replacer;
  hasFailure = false;
  auto paramTypeReplacement = [&](ParamClosureType type) -> Type {
    auto ptr = lifter.paramClosureTypeToType.find(type);
    if (ptr != lifter.paramClosureTypeToType.end())
      return ptr->second;
    mlir::emitError(theModule.getLoc())
        << "no type found for paramclosure type " << type;
    hasFailure = true;
    return type;
  };
  mlir::AttrTypeReplacer generatorReplacer;
  generatorReplacer.addReplacement([&](ClosureType type) -> Type {
    auto ptr = lifter.closureTypeToStructTypes.find(type);
    if (ptr != lifter.closureTypeToStructTypes.end())
      return ptr->second;
    return type;
  });
  generatorReplacer.addReplacement([&](ClosureAttr attr) -> Attribute {
    auto ptr = lifter.paramCaptureToStructAttr.find(attr);
    if (ptr != lifter.paramCaptureToStructAttr.end())
      return ptr->second;
    mlir::emitError(theModule.getLoc())
        << "no capture struct attr found for closure attr " << attr;
    hasFailure = true;
    return attr;
  });
  generatorReplacer.addReplacement(paramTypeReplacement);
  generatorReplacer.recursivelyReplaceElementsIn(enclosingOp, true, true, true);

  for (auto [closureType, structGenerator] : structGenerators) {
    SmallVector<ParamDeclAttr> newParams;
    for (auto param : structGenerator.getInputParams()) {
      if (auto abstractType = dyn_cast<ParamClosureType>(param.getType())) {
        newParams.push_back(ParamDeclAttr::get(
            param.getName(), paramTypeReplacement(abstractType)));
        continue;
      }
      newParams.push_back(param);
    }
    structGenerator.setInputParams(newParams);
    structGenerator.walk([&](WitnessOp w) {
      if (auto sym = dyn_cast<ClosureSymbolAttr>(w.getValue())) {
        auto it = lifter.liftedClosureSymbols.find(sym);
        assert(
            it != lifter.liftedClosureSymbols.end() &&
            "should not be possible to reach here unless the front end "
            "compiled a conformance table but then did not store the symbols "
            "necessary to generate the function the witness points to.");
        w.setValueAttr(it->second);
      }
    });

    auto packedClosureType = lifter.packedClosureType[closureType];
    assert(packedClosureType && "expected a replaced closure type");
    structGenerator.setValueDomainType(packedClosureType);
  }
  return hasFailure ? failure() : success();
}

// lift closures and replace closure.init
void OutlineClosuresNewPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  for (auto generator : theModule.getOps<GeneratorOp>()) {
    bool hasFailure = false;
    generator.walk<mlir::WalkOrder::PostOrder>([&](Operation *operation) {
      if (!isa<GeneratorOp, ClosureInitOp>(operation))
        return WalkResult::advance();
      if (failed(liftClosuresFromRegion(theModule, symtab, paramCache,
                                        debugBuild, operation))) {
        hasFailure = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasFailure)
      return signalPassFailure();
  }
}
