//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CODialect/COOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace CO;

//===----------------------------------------------------------------------===//
// Lower Async Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERASYNCFUNCTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerAsyncFunctionsPass
    : impl::LowerAsyncFunctionsBase<LowerAsyncFunctionsPass> {
public:
  using LowerAsyncFunctionsBase::LowerAsyncFunctionsBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    coroAttrName = StringAttr::get(ctx, "coro");
    return success();
  }
  void runOnOperation() override;

private:
  StringAttr coroAttrName;
};
} // namespace

enum ContinuationField {
  State = 0,
  ResumeFunction = 1,
  CallbackFn = 2,
  ClosureState = 3,
  Frame = 4,
  Promise = 5
};

struct COTypeConverter {
  Type typeForField(ContinuationField field) {
    switch (field) {
    case State:
      return IndexType::get(cxt);
    case ResumeFunction:
      return resumeSignatureType;
    case CallbackFn:
      return callbackSignature;
    case Frame:
    case Promise:
    case ClosureState:
      return opaquePointerType;
    }
  }
  COTypeConverter(MLIRContext *cxt);
  COTypeConverter(const COTypeConverter &) = delete;
  COTypeConverter &operator=(const COTypeConverter &) = delete;

public:
  Type convertType(Type type) const {
    if (isa<CO::CoroutineType>(type))
      return PointerType::get(continuationType);
    ;
    return type;
  }
  Type getContinuationType() const { return continuationType; }

private:
  Type continuationType;
  Type resumeSignatureType;
  Type opaquePointerType;
  Type callbackSignature;
  MLIRContext *cxt;
};

struct LoweredAsyncFunction {
  LoweredAsyncFunction(FuncOp ramp, FuncOp resume)
      : ramp(ramp), resume(resume) {}
  FuncOp ramp;
  FuncOp resume;
};

struct LowerAsyncBuildContext {
  LowerAsyncBuildContext(StringAttr coroAttrName,
                         COTypeConverter &coTypeConverter,
                         Shared<SymbolTable &> &sharedTable,
                         DenseMap<SymbolConstantAttr, SymbolConstantAttr>
                             &asyncFuncToRampFunctions,
                         ImplicitLocOpBuilder &builder)
      : coroAttrName(coroAttrName), coTypeConverter(coTypeConverter),
        sharedTable(sharedTable),
        asyncFuncToRampFunctions(asyncFuncToRampFunctions), builder(builder) {}
  void populateRampFunction(FuncOp rampFunction, FuncOp resumeFunction,
                            Type frameType) {
    Type continuationType = coTypeConverter.getContinuationType();
    builder.setInsertionPointToStart(
        &rampFunction.getBodyRegion().emplaceBlock());
    for (Type argument : rampFunction.getSignature().getArguments())
      rampFunction.getBodyRegion().addArgument(argument, rampFunction.getLoc());
    // Allocate memory for continuation.
    TypedAttr target = ParamOperatorAttr::get(POC::CurrentTarget, {},
                                              builder.getType<TargetType>());
    TypedAttr elementType = TypeConstantAttr::get(
        continuationType, TypeType::get(continuationType.getContext()));
    Value sizeOf = builder.create<ParamConstantOp>(
        ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
    Value alignOf = builder.create<ParamConstantOp>(
        ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
    Value continuation = builder.create<AlignedAllocOp>(
        PointerType::get(continuationType), ValueRange{alignOf, sizeOf});
    // Store resume function.
    Value resumeFunctionSlot =
        builder.create<StructGEPOp>(continuation, ResumeFunction);
    Value functionPointer = builder.create<CreateClosureOp>(
        SymbolConstantAttr::get(SymbolRefAttr::get(builder.getContext(),
                                                   resumeFunction.getSymName()),
                                resumeFunction.getSignature()));
    builder.create<StoreOp>(functionPointer, resumeFunctionSlot);

    // Store arguments in frame.
    Value frameSlot = builder.create<StructGEPOp>(continuation, Frame);
    Value frameOpaque = builder.create<LoadOp>(frameSlot);
    Value frame = builder.create<PointerBitcastOp>(PointerType::get(frameType),
                                                   frameOpaque);
    for (BlockArgument arg : rampFunction.getArguments()) {
      Value slot = builder.create<StructGEPOp>(frame, arg.getArgNumber());
      builder.create<StoreOp>(arg, slot);
    }
    builder.create<ReturnOp>(continuation);
  }

  void populateResumeFunction(FuncOp resumeFunction, FuncOp funcOp,
                              Type frameType) {
    Type continuationType = coTypeConverter.getContinuationType();
    // Take the body of the original function.
    resumeFunction.getBodyRegion().takeBody(funcOp.getBodyRegion());
    resumeFunction.getBodyRegion().insertArgument(
        (unsigned)0, resumeFunction.getSignature().getArguments()[0],
        resumeFunction->getLoc());
    builder.setInsertionPointToStart(&resumeFunction.getBodyRegion().front());

    // Extract arguments from continuation's frame.
    Value continuation = builder.create<PointerBitcastOp>(
        PointerType::get(continuationType), resumeFunction.getArgument(0));
    Value frameSlot = builder.create<StructGEPOp>(continuation, Frame);
    Value frameOpaque = builder.create<LoadOp>(frameSlot);
    Value frame = builder.create<PointerBitcastOp>(PointerType::get(frameType),
                                                   frameOpaque);

    // Map arguments of func to values extracted from continuation frame.
    for (BlockArgument argument :
         resumeFunction.getBodyRegion().front().getArguments().slice(1)) {
      Value extractedValue =
          builder.create<StructGEPOp>(frame, argument.getArgNumber() - 1);
      Value loadedValue = builder.create<LoadOp>(extractedValue);
      argument.replaceAllUsesWith(loadedValue);
    }
    llvm::BitVector args(
        resumeFunction.getBodyRegion().front().getNumArguments(), true);
    args.reset(0);
    resumeFunction.getBodyRegion().front().eraseArguments(args);

    // Replace ReturnOps with set result.
    resumeFunction.walk([&](ReturnOp returnOp) {
      if (returnOp.getNumOperands() > 0) {
        builder.setInsertionPoint(returnOp);
        Value promiseSlot = builder.create<StructGEPOp>(continuation, Promise);
        Value promise = builder.create<LoadOp>(promiseSlot);
        Value typedPromise = builder.create<PointerBitcastOp>(
            PointerType::get(funcOp.getSignature().getResults()[0]), promise);
        builder.create<StoreOp>(returnOp.getOperand(0), typedPromise);
        builder.create<ReturnOp>();
        returnOp->erase();
      }
    });
  }

  LoweredAsyncFunction lowerAsyncFunction(FuncOp funcOp) {
    MLIRContext *cxt = funcOp.getContext();
    Type continuationType = coTypeConverter.getContinuationType();

    // Create Ramp Function.
    StringAttr rampName = builder.getStringAttr(funcOp.getSymName() + "_ramp");
    FunctionType rampFunctionType =
        builder.getFunctionType(funcOp.getBodyRegion().getArgumentTypes(),
                                PointerType::get(continuationType));
    auto rampSignature = SignatureType::get(rampFunctionType);
    builder.setInsertionPoint(funcOp);
    FuncOp rampFunction = builder.create<FuncOp>(rampName, rampSignature);
    rampName = sharedTable.modify(
        [rampFunction, it = funcOp->getIterator()](SymbolTable &symtab) {
          return symtab.insert(rampFunction, it);
        });

    // Create resume function.
    StringAttr resumeName =
        builder.getStringAttr(funcOp.getSymName() + "_resume");
    Type opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
    SmallVector<Type> inputs;
    SmallVector<Type> results;
    inputs.push_back(opaquePointerType);
    FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
    auto resumeSignature = SignatureType::get(resumeFunctionType);
    FuncOp resumeFunction = builder.create<FuncOp>(
        funcOp->getParentOp()->getLoc(), resumeName, resumeSignature);
    resumeFunction->setAttr(coroAttrName, mlir::UnitAttr::get(cxt));
    resumeName = sharedTable.modify(
        [resumeFunction, it = rampFunction->getIterator()](
            SymbolTable &symtab) { return symtab.insert(resumeFunction, it); });

    // TODO: Calculate Frame.
    SmallVector<Type> frameTypes;
    for (Type argumentType : funcOp.getArgumentTypes())
      frameTypes.push_back(argumentType);
    Type frameType = StructType::get(frameTypes);
    populateRampFunction(rampFunction, resumeFunction, frameType);
    populateResumeFunction(resumeFunction, funcOp, frameType);

    // Update map.
    SymbolConstantAttr key = SymbolConstantAttr::get(
        SymbolRefAttr::get(funcOp.getContext(), funcOp.getSymName()),
        funcOp.getSignature());
    SymbolConstantAttr value = SymbolConstantAttr::get(
        SymbolRefAttr::get(cxt, rampFunction.getSymName()),
        rampFunction.getSignature());
    asyncFuncToRampFunctions[key] = value;
    funcOp.erase();
    return {rampFunction, resumeFunction};
  }

private:
  StringAttr coroAttrName;
  COTypeConverter &coTypeConverter;
  Shared<SymbolTable &> &sharedTable;
  DenseMap<SymbolConstantAttr, SymbolConstantAttr> &asyncFuncToRampFunctions;
  ImplicitLocOpBuilder &builder;
};

COTypeConverter::COTypeConverter(MLIRContext *cxt) : cxt(cxt) {
  opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.push_back(opaquePointerType);
  FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
  resumeSignatureType = SignatureType::get(resumeFunctionType);
  FunctionType callbackFunctionType =
      FunctionType::get(cxt, opaquePointerType, KGEN::NoneType::get(cxt));
  callbackSignature = SignatureType::get(callbackFunctionType);

  // Build Continuation Type.
  std::array<Type, 6> types;
  types[State] = typeForField(State);
  types[ResumeFunction] = typeForField(ResumeFunction);
  types[CallbackFn] = typeForField(CallbackFn);
  types[ClosureState] = typeForField(ClosureState);
  types[Frame] = typeForField(Frame);
  types[Promise] = typeForField(Promise);
  continuationType = StructType::get(types);
}

void LowerAsyncFunctionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  COTypeConverter typeConverter(module.getContext());
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  Shared<SymbolTable &> sharedTable(symtab);
  DenseMap<SymbolConstantAttr, SymbolConstantAttr> asyncFuncToRampFunctions;

  // Convert async functions.
  ImplicitLocOpBuilder b(module->getLoc(), module);
  Operation *op = &*module.getOps().begin();
  LowerAsyncBuildContext buildContext(coroAttrName, typeConverter, sharedTable,
                                      asyncFuncToRampFunctions, b);
  while (op) {
    Operation *next = op->getNextNode();
    if (FuncOp funcOp = dyn_cast<FuncOp>(op)) {
      if (funcOp.isAsync())
        buildContext.lowerAsyncFunction(funcOp);
    }
    op = next;
  }

  // Apply all other CO lowerings.
  mlir::IRRewriter rewriter(b);
  module.walk([&](Operation *op) {
    if (auto invokeOp = dyn_cast<InvokeOp>(op)) {
      auto symbol = cast<SymbolConstantAttr>(invokeOp.getCallee());
      auto newSymbolPtr = asyncFuncToRampFunctions.find(symbol);
      if (newSymbolPtr != asyncFuncToRampFunctions.end()) {
        SymbolConstantAttr newSymbol = newSymbolPtr->getSecond();
        Type continuationType =
            typeConverter.convertType(invokeOp->getResultTypes().front());
        rewriter.setInsertionPoint(op);
        auto callOp =
            rewriter.create<CallOp>(invokeOp->getLoc(), continuationType,
                                    newSymbol, invokeOp.getOperands());
        rewriter.replaceOp(invokeOp, callOp);
      }
    }
  });
}
