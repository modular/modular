//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLVMLoweringUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERPOPCLOSURESTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPPartialApply
//===----------------------------------------------------------------------===//

class ConvertPOPPartialApply
    : public mlir::ConvertOpToLLVMPattern<PartialApplyOp> {
public:
  ConvertPOPPartialApply(SymbolTable &symtab,
                         mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter), symtab(symtab) {}

  LogicalResult
  matchAndRewrite(PartialApplyOp op, PartialApplyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType calleeType;
    ClosureType calleeClosureType =
        dyn_cast<ClosureType>(op.getCallee().getType());
    if (calleeClosureType)
      calleeType = calleeClosureType.getFunc();
    else
      calleeType = dyn_cast<FunctionType>(op.getCallee().getType());

    // Create the type of the "environment" struct
    size_t numBoundArgs = op.getBoundInputs().size();
    size_t numCalleeInputs = calleeType.getInputs().size();

    SmallVector<Type> boundArgTypes;
    boundArgTypes.reserve(numBoundArgs);

    for (Value arg : op.getInputs()) {
      Type ty = getTypeConverter()->convertType(arg.getType());
      if (!ty)
        return emitError(op.getLoc())
               << "could not convert bound argument type for argument " << arg
               << " with type " << arg.getType();
      boundArgTypes.push_back(ty);
    }
    auto erasedPtrType = LLVM::LLVMPointerType::get(context);
    auto erasedEnvPtrType =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
            context, {erasedPtrType, erasedPtrType}));

    if (calleeClosureType) {
      // Add the type of the parent closure's environment struct
      boundArgTypes.push_back(erasedEnvPtrType);
    }
    auto boundArgStructTy =
        LLVM::LLVMStructType::getLiteral(context, boundArgTypes);
    auto envStructType = LLVM::LLVMStructType::getLiteral(
        context, {erasedPtrType, boundArgStructTy});
    auto envStructPtrTy = LLVM::LLVMPointerType::get(envStructType);

    // Create the wrapper function type
    // Wrapper function has the type (!llvm.ptr, unboundArgTy0, ...
    // unboundArgTyn) -> resultTy0, ... resultTym
    SmallVector<Type> wrapperFnArgTypes;
    wrapperFnArgTypes.push_back(erasedPtrType);

    // Add converted types of the unbound arguments to the arguments for the
    // new function
    ArrayRef<Type> closureFuncInputTys = op.getType().getFunc().getInputs();
    for (Type inpTy : closureFuncInputTys) {
      Type ty = getTypeConverter()->convertType(inpTy);
      if (!ty)
        return emitError(op.getLoc()) << "could not convert type " << inpTy;
      wrapperFnArgTypes.push_back(ty);
    }

    // Convert the result types.
    SmallVector<Type> resultTypes;
    Type packedResTy = getVoidType();
    if (calleeType.getNumResults()) {
      packedResTy =
          getTypeConverter()->packFunctionResults(calleeType.getResults());
      resultTypes.assign({packedResTy});
      if (!resultTypes.back())
        return emitError(op.getLoc(), "failed to convert call result types");
    }

    auto wrapperFnType = LLVM::LLVMFunctionType::get(
        context, packedResTy, wrapperFnArgTypes, /*isVarArg*/ false);

    // Create the body of the wrapper function
    Block *wrapperFnBody = new Block;
    Block *oldInsertionBlock = rewriter.getInsertionBlock();
    Block::iterator oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPointToStart(wrapperFnBody);

    // Add wrapper function arguments to the block arguments
    for (Type argTy : wrapperFnArgTypes)
      wrapperFnBody->addArgument(argTy, op.getLoc());

    Value env = wrapperFnBody->getArgument(0);
    Value envStructPtr =
        rewriter.create<LLVM::BitcastOp>(op.getLoc(), envStructPtrTy, env);

    Type envCalleeType;
    if (calleeClosureType) {
      SmallVector<Type> inpTypes;
      inpTypes.push_back(erasedEnvPtrType);
      for (Type inpType : calleeType.getInputs()) {
        Type ty = getTypeConverter()->convertType(inpType);
        if (!ty)
          return emitError(op.getLoc())
                 << "could not convert input type " << inpType;
        inpTypes.push_back(ty);
      }

      // If there are no result types, set it to void. Otherwise, set the result
      // type to the packed result types.
      Type resultType;
      if (resultTypes.empty())
        resultType = getVoidType();
      else
        resultType = resultTypes[0];

      envCalleeType = LLVM::LLVMPointerType::get(
          LLVM::LLVMFunctionType::get(context, resultType, inpTypes, 0));
    } else {
      envCalleeType = adaptor.getCallee().getType();
    }

    // This implicitly bitcasts the pointer to a function
    Value calleePtr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(envCalleeType), envStructPtr,
        ArrayRef<LLVM::GEPArg>({0, 0}));

    Value callee = rewriter.create<LLVM::LoadOp>(op.getLoc(), calleePtr);

    // Create the call to the callee
    SmallVector<Value> llvmCallArgs;
    llvmCallArgs.push_back(callee);

    if (calleeClosureType) {
      // The parent closure's environment struct is at the end of the bound
      // arguments
      Value boundEnvPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(erasedEnvPtrType),
          envStructPtr, ArrayRef<LLVM::GEPArg>({0, 1, numBoundArgs}));
      Value boundEnv = rewriter.create<LLVM::LoadOp>(op.getLoc(), boundEnvPtr);
      llvmCallArgs.push_back(boundEnv);
    }

    // Unpack the bound and unbound arguments into the call args
    size_t boundArgIdx = 0;
    // The environment is the 0 argument to wrapperFn.
    size_t wrapperFnArgIdx = 1;
    ArrayRef<int64_t> boundInputs = op.getBoundInputs();
    for (size_t i = 0; i < numCalleeInputs; i++) {
      if (boundArgIdx < numBoundArgs &&
          i == static_cast<size_t>(boundInputs[boundArgIdx])) {
        // Extract the bound argument from the env struct and add it to
        // llvmCallArgs
        Type boundArgType = boundArgTypes[boundArgIdx];
        LLVM::LLVMPointerType boundArgPtrType =
            LLVM::LLVMPointerType::get(boundArgType);
        Value boundArgPtr = rewriter.create<LLVM::GEPOp>(
            op.getLoc(), boundArgPtrType, envStructPtr,
            ArrayRef<LLVM::GEPArg>({0, 1, boundArgIdx}));
        Value boundArg = rewriter.create<LLVM::LoadOp>(
            op.getLoc(), boundArgType, boundArgPtr);
        llvmCallArgs.push_back(boundArg);
        ++boundArgIdx;
      } else {
        // Add the wrapper function argument to llvmCallArgs
        llvmCallArgs.push_back(wrapperFnBody->getArgument(wrapperFnArgIdx));
        ++wrapperFnArgIdx;
      }
    }

    LLVM::CallOp callWithUnpackedArgs = rewriter.create<LLVM::CallOp>(
        op.getLoc(), resultTypes, FlatSymbolRefAttr(), llvmCallArgs);
    rewriter.create<LLVM::ReturnOp>(op.getLoc(),
                                    callWithUnpackedArgs.getResults());

    // Insert wrapper function into the module and reset the insertion point
    // to the partial apply op
    rewriter.clearInsertionPoint();
    LLVM::LLVMFuncOp wrapperFn = rewriter.create<LLVM::LLVMFuncOp>(
        op.getLoc(), "closure_wrapper_fn", wrapperFnType);
    wrapperFn.getBody().push_back(wrapperFnBody);

    symtab.insert(wrapperFn);

    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);

    // Create the struct representing the closure
    LLVM::LLVMStructType closureStructType = LLVM::LLVMStructType::getLiteral(
        context, {erasedPtrType, erasedPtrType});
    Value closureStruct =
        rewriter.create<LLVM::UndefOp>(op.getLoc(), closureStructType);

    // Put the pointer to the wrapper function into the closure struct
    Value wrapperFnPtr =
        rewriter.create<LLVM::AddressOfOp>(op.getLoc(), wrapperFn);
    Value erasedWrapperFnPtr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), erasedPtrType, wrapperFnPtr);
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, erasedWrapperFnPtr, 0);

    // Allocate the env struct
    Value one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(context, 8), 1);
    Value envStruct =
        rewriter.create<LLVM::AllocaOp>(op.getLoc(), envStructPtrTy, one);
    // TODO: When data layouts are propagated properly, extract the data
    // layout from TargetInfoAttr
    mlir::DataLayout defaultDL = mlir::DataLayout();
    size_t envSize = defaultDL.getTypeSize(envStructType);

    // Insert lifetime marker for the env struct
    rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), envSize, envStruct);

    // Add the bound arguments to the environment struct
    for (auto [argIdx, boundArgValue] : llvm::enumerate(adaptor.getInputs())) {
      Type boundArgTy = boundArgTypes[argIdx];
      Value boundArgPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(boundArgTy), envStruct,
          ArrayRef<LLVM::GEPArg>({0, 1, argIdx}));
      rewriter.create<LLVM::StoreOp>(op.getLoc(), boundArgValue, boundArgPtr);
    }
    if (calleeClosureType) {
      Value parentEnvPtr = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 1);
      Value boundParentEnvPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(erasedPtrType), envStruct,
          ArrayRef<LLVM::GEPArg>({0, 1, numBoundArgs}));
      rewriter.create<LLVM::StoreOp>(op.getLoc(), parentEnvPtr,
                                     boundParentEnvPtr);
    }

    // Add the pointer to the original callee to the environment struct
    Value calleePtrValue;

    if (calleeClosureType) {
      calleePtrValue = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 0);
      calleePtrValue = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), envCalleeType, calleePtrValue);
    } else {
      calleePtrValue = adaptor.getCallee();
    }
    Value envCalleePtr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(envCalleeType), envStruct,
        ArrayRef<LLVM::GEPArg>({0, 0}));
    rewriter.create<LLVM::StoreOp>(op.getLoc(), calleePtrValue, envCalleePtr);
    // Add the environment struct to the closure struct
    LLVM::BitcastOp erasedEnvStructPtr =
        rewriter.create<LLVM::BitcastOp>(op.getLoc(), erasedPtrType, envStruct);
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, erasedEnvStructPtr, 1);

    // Insert lifetime marker at the end of the struct
    oldInsertionBlock = rewriter.getInsertionBlock();
    oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), envSize, envStruct);
    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);

    rewriter.replaceOp(op, closureStruct);
    return success();
  }

private:
  MLIRContext *context = &this->getTypeConverter()->getContext();
  SymbolTable &symtab;
};

//===----------------------------------------------------------------------===//
// ConvertPOPCallIndirect
//===----------------------------------------------------------------------===//

struct ConvertPOPCallIndirect : mlir::ConvertOpToLLVMPattern<CallIndirectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallIndirectOp op, CallIndirectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result types.
    SmallVector<Type> resultTypes;
    if (op.getNumResults()) {
      resultTypes.assign(
          {getTypeConverter()->packFunctionResults(op.getResultTypes())});
      if (!resultTypes.back())
        return emitError(op.getLoc(), "failed to convert call result types");
    }
    Type resultType;
    // If there are no result types, set it to void. Otherwise, set the result
    // type to the packed result types.
    if (resultTypes.empty())
      resultType = getVoidType();
    else
      resultType = resultTypes[0];

    Value callee = op.getCallee();
    LLVM::CallOp llvmCall;
    if (isa<ClosureType>(callee.getType())) {
      // Unpack the struct representation of the closure.
      auto pointerType = LLVM::LLVMPointerType::get(context);
      Value wrapperFnPtr = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 0);
      Value envStruct = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 1);

      // Compute the type of the wrapper function -- wrapper function type is
      // (!llvm.ptr, unboundArgTy0, ... unboundArgTyn) -> resultTypes
      SmallVector<Type> wrapperFnArgTypes;
      wrapperFnArgTypes.push_back(pointerType);

      auto calleeFuncTy = callee.getType().dyn_cast<ClosureType>().getFunc();
      for (Type argTy : calleeFuncTy.getInputs()) {
        Type ty = getTypeConverter()->convertType(argTy);
        if (!ty)
          return emitError(op.getLoc())
                 << "could not convert argument type " << argTy;
        wrapperFnArgTypes.push_back(ty);
      }

      auto wrapperFnType = LLVM::LLVMFunctionType::get(context, resultType,
                                                       wrapperFnArgTypes, 0);
      Value castWrapperFn = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(wrapperFnType), wrapperFnPtr);

      // Create the call to the wrapper function.
      SmallVector<Value> llvmCallArgs;
      llvmCallArgs.push_back(castWrapperFn);
      llvmCallArgs.push_back(envStruct);
      for (Value inp : adaptor.getInputs())
        llvmCallArgs.push_back(inp);

      llvmCall = rewriter.create<LLVM::CallOp>(
          op.getLoc(), resultTypes, FlatSymbolRefAttr(), llvmCallArgs);
    } else {
      // Create the LLVM call operation.
      // Note: adaptor.getOperands() is a list of callee followed by inputs.
      llvmCall = rewriter.create<LLVM::CallOp>(
          op.getLoc(), resultTypes, FlatSymbolRefAttr(), adaptor.getOperands());
    }

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, llvmCall.getResults());
      return success();
    }

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), llvmCall.getResult(), i));

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  MLIRContext *context = &this->getTypeConverter()->getContext();
};

} //  namespace

//===----------------------------------------------------------------------===//
// LowerPOPClosuresToLLVMPass
//===----------------------------------------------------------------------===//

/// Closures are represented by a struct containing {wrapperFnPtr,
/// envStructPtr}. The struct is type-erased, so it has type {!llvm.ptr,
/// !llvm.ptr}. The envStruct contains the bound arguments and a pointer to
/// the original callee, in the form
/// {originalCalleePtr, {boundArg0, .. boundArgn}},
/// and it is allocated on the stack. The wrapper function takes in the
/// environment struct followed by arguments which are not bound.
/// WARNING: Since closures are stack allocated, you cannot return
/// a closure into a higher memory scope.
///
/// For example,
/// my_fn(a: A, b: B, c: C, d: D) -> E
/// my_closure = my_fn(a0, ?, c0, ?)

/// becomes:

/// {wrapperFn*, env*}, with wrapperFn and env defined below:
/// wrapperFn(env: Env, b: B, d: D) -> E {
///   my_fn* = env[0]
///   bound_args = env[1]
///   return my_fn(bound_args[0], b, bound_args[1], d)
/// }
/// env = {*my_fn, {a0, c0}}

struct LowerPOPClosuresToLLVMPass
    : public KGEN::impl::LowerPOPClosuresToLLVMBase<
          LowerPOPClosuresToLLVMPass> {
  using LowerPOPClosuresToLLVMBase::LowerPOPClosuresToLLVMBase;

  void runOnOperation() override;
};

void LowerPOPClosuresToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Configure the type converter.
  FailureOr<mlir::LowerToLLVMOptions> options =
      getTargetLoweringOptions(theModule);
  if (failed(options))
    return signalPassFailure();
  POPToLLVMTypeConverter typeConverter(theModule.getLoc(), *options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Convert partial apply ops.
  target.addIllegalOp<PartialApplyOp>();
  patterns.insert<ConvertPOPPartialApply>(symtab, typeConverter);

  // Convert call indirect ops.
  target.addIllegalOp<CallIndirectOp>();
  patterns.insert<ConvertPOPCallIndirect>(typeConverter);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
