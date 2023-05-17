//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

#define DEBUG_TYPE "lower-runtime-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERRUNTIMECLOSURES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerRuntimeClosuresPass
    : M::KGEN::impl::LowerRuntimeClosuresBase<LowerRuntimeClosuresPass> {
  void runOnOperation() override;
};
} // namespace

struct CreateClosureTypes {
public:
  CreateClosureTypes() {}
  static LogicalResult
  createClosureTypes(CreateClosureTypes &types, CreateClosureOp op,
                     mlir::LLVMTypeConverter &typeConverter);
  SmallVector<Type> boundArgTypes;
  Type opaquePtrType;
  /// Two element struct where first element is void* to function
  /// and second member is struct of captured elements.
  Type liftedFunctionCaptureType;
  Type liftedFunctionCaptureTypePtrType;
  /// The opaque closure struct type is a struct of two opaque pointers
  /// The first element points to a function interface method.
  /// The second points to captured state + lifted function.
  Type unpackingFunctionAndCapturesType;
};

LogicalResult
CreateClosureTypes::createClosureTypes(CreateClosureTypes &types,
                                       CreateClosureOp op,
                                       mlir::LLVMTypeConverter &typeConverter) {
  MLIRContext *context = op.getContext();
  types.opaquePtrType = LLVM::LLVMPointerType::get(context);
  types.boundArgTypes.reserve(op.getCaptures().size());
  for (Value arg : op.getCaptures()) {
    Type ty = typeConverter.convertType(arg.getType());
    if (!ty)
      return failure();
    types.boundArgTypes.push_back(ty);
  }
  types.liftedFunctionCaptureType =
      LLVM::LLVMStructType::getLiteral(context, types.boundArgTypes);
  types.liftedFunctionCaptureTypePtrType =
      LLVM::LLVMPointerType::get(types.liftedFunctionCaptureType);
  types.unpackingFunctionAndCapturesType = LLVM::LLVMStructType::getLiteral(
      context, {types.opaquePtrType, types.opaquePtrType});
  return success();
}

struct CreateRuntimeClosureOpConversion
    : public ConvertPOPToLLVMPattern<CreateClosureOp> {

  CreateRuntimeClosureOpConversion(SymbolTable &symTable,
                                   POPToLLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern<CreateClosureOp>(typeConverter),
        symtab(symTable) {}

private:
  SymbolTable &symtab;

  LLVM::LLVMFuncOp
  generateWrapperFunction(CreateClosureOp op,
                          ConversionPatternRewriter &rewriter) const {
    SignatureType calleeSignature = op.getCalleeType();
    FunctionType calleeType = calleeSignature.getValues();
    MLIRContext *context = getContext();

    // the signature if the wrapper is (opaquePointer, PM, ..., PN) -> R, where
    // M is the number of captures and N is the number of arguments of the
    // lifted function
    Type packedResTy = LLVM::LLVMVoidType::get(context);
    if (calleeType.getNumResults())
      packedResTy =
          getTypeConverter()->packFunctionResults(calleeType.getResults());

    // Create input types
    auto opaquePtrType = LLVM::LLVMPointerType::get(context);
    SmallVector<Type> wrapperFnArgTypes;
    wrapperFnArgTypes.push_back(opaquePtrType);
    size_t captureCount = op.getCaptures().size();
    ArrayRef<Type> noncapturedInputs = calleeType.getInputs().slice(
        captureCount, calleeType.getNumInputs() - captureCount);
    for (Type inpTy : noncapturedInputs) {
      Type ty = convertType(inpTy);
      if (!ty)
        return {};
      wrapperFnArgTypes.push_back(ty);
    }
    auto wrapperFnType = LLVM::LLVMFunctionType::get(
        context, packedResTy, wrapperFnArgTypes, /*isVarArg*/ false);

    // Create the body of the wrapper function
    Block *wrapperFnBody = new Block;
    rewriter.setInsertionPointToStart(wrapperFnBody);

    // Add wrapper function arguments to the block arguments
    for (Type argTy : wrapperFnArgTypes)
      wrapperFnBody->addArgument(argTy, op.getLoc());
    rewriter.clearInsertionPoint();
    auto wrapperFn = rewriter.create<LLVM::LLVMFuncOp>(
        op.getLoc(), "closure_wrapper_fn", wrapperFnType,
        LLVM::Linkage::Internal);
    wrapperFn.getBody().push_back(wrapperFnBody);
    return wrapperFn;
  }

  /// The body of the closure wrapper function should consist of casting the
  /// opaque pointer to the captured state and unpacking the members in order to
  /// call the original lifted function.
  LogicalResult populateBodyOfWrapperFunction(
      LLVM::LLVMFuncOp wrapperFn, CreateClosureOp op,
      ConversionPatternRewriter &rewriter, CreateClosureOpAdaptor adaptor,
      SymbolTable &symbolTable, CreateClosureTypes const &types) const {
    Block &wrapperFnBody = wrapperFn.getBody().front();
    rewriter.setInsertionPointToStart(&wrapperFnBody);
    Value envStructPtr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), types.liftedFunctionCaptureTypePtrType,
        wrapperFnBody.getArgument(0));

    Type envCalleeType = adaptor.getCallee().getType();
    if (auto sigType = dyn_cast<SignatureType>(envCalleeType))
      envCalleeType = typeConverter->convertType(sigType.getValues());

    SmallVector<Value> liftedNestedFunctionCallArgs(
        op.getCalleeType().getValues().getNumInputs());
    auto flatSymbol = dyn_cast<FlatSymbolRefAttr>(
        cast<SymbolConstantAttr>(op.getCallee()).getSymbol());
    if (!flatSymbol)
      return emitError(op.getLoc(),
                       "cannot lower call to nested symbol to LLVM");
    auto llvmFunction = symbolTable.lookup(flatSymbol.getRootReference());
    auto func = dyn_cast<LLVM::LLVMFuncOp>(llvmFunction);
    if (!func)
      return emitError(op.getLoc(), "Callee does not reference llvm function");

    for (size_t i = 0; i < op.getCaptures().size(); i++) {
      Type capturedArgType = types.boundArgTypes[i];
      LLVM::LLVMPointerType boundArgPtrType =
          LLVM::LLVMPointerType::get(capturedArgType);
      Value boundArgPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), boundArgPtrType, envStructPtr,
          ArrayRef<LLVM::GEPArg>({0, i}));
      Value boundArg = rewriter.create<LLVM::LoadOp>(
          op.getLoc(), capturedArgType, boundArgPtr);
      liftedNestedFunctionCallArgs[i] = boundArg;
    }
    size_t numCaptures = op.getCaptures().size();
    size_t numberDynamicArgs =
        op.getCalleeType().getValues().getNumInputs() - numCaptures;
    for (size_t i = 0; i < numberDynamicArgs; i++)
      liftedNestedFunctionCallArgs[i + numCaptures] =
          wrapperFnBody.getArgument(i + 1);

    ValueRange valueRange(liftedNestedFunctionCallArgs);
    auto callLiftedFunction =
        rewriter.create<LLVM::CallOp>(op.getLoc(), func, valueRange);
    rewriter.create<LLVM::ReturnOp>(op.getLoc(),
                                    callLiftedFunction.getResults());
    return success();
  }

  /// Replace the CreateClosureOp with the construction of a closure struct.
  LogicalResult generateClosureStruct(ConversionPatternRewriter &rewriter,
                                      CreateClosureOp op,
                                      CreateClosureOpAdaptor adaptor,
                                      LLVM::LLVMFuncOp wrapperFn,
                                      CreateClosureTypes const &types) const {
    MLIRContext *context = getContext();
    Value closureStruct = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), types.unpackingFunctionAndCapturesType);
    Value addressOfWrapperFunction =
        rewriter.create<LLVM::AddressOfOp>(op.getLoc(), wrapperFn);

    LLVM::BitcastOp erasedEnvStructPtr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), types.opaquePtrType, addressOfWrapperFunction);
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, erasedEnvStructPtr, 0);
    Value one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(context, 8), 1);
    Value envStruct = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), types.liftedFunctionCaptureTypePtrType, one);
    // TODO: When data layouts are propagated properly, extract the data
    //  layout from TargetInfoAttr
    size_t envSize =
        getTypeConverter()->getTypeAllocSize(types.liftedFunctionCaptureType);

    rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), envSize, envStruct);
    for (auto [argIdx, boundArgValue] :
         llvm::enumerate(adaptor.getCaptures())) {
      Type boundArgTy = types.boundArgTypes[argIdx];
      Value boundArgPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(boundArgTy), envStruct,
          ArrayRef<LLVM::GEPArg>({0, argIdx}));
      rewriter.create<LLVM::StoreOp>(op.getLoc(), boundArgValue, boundArgPtr);
    }

    // Add the environment struct to the closure struct
    LLVM::BitcastOp erasedEnvStructPtr2 = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), types.opaquePtrType, envStruct);
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, erasedEnvStructPtr2, 1);

    // Insert lifetime marker at the end of the struct
    auto oldInsertionBlock = rewriter.getInsertionBlock();
    auto oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), envSize, envStruct);
    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);
    rewriter.replaceOp(op, closureStruct);

    return success();
  }

public:
  LogicalResult
  matchAndRewrite(CreateClosureOp op, CreateClosureOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isCapturing() && op.getCaptures().empty()) {
      rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
          op, convertType(op.getType()),
          cast<FlatSymbolRefAttr>(
              cast<SymbolConstantAttr>(op.getCallee()).getSymbol()));
      return success();
    }

    // Generate the function wrapper and populate it with extract and invoke the
    // original callee
    Block *oldInsertionBlock = rewriter.getInsertionBlock();
    Block::iterator oldInsertionPoint = rewriter.getInsertionPoint();
    rewriter.clearInsertionPoint();
    LLVM::LLVMFuncOp wrapperFn = this->generateWrapperFunction(op, rewriter);
    StringAttr name = symtab.insert(wrapperFn);
    CreateClosureTypes types;
    if (failed(CreateClosureTypes::createClosureTypes(types, op,
                                                      *getTypeConverter())))
      return emitError(op.getLoc(),
                       "failed to convert kgen types to llvm closure types");

    if (failed(this->populateBodyOfWrapperFunction(wrapperFn, op, rewriter,
                                                   adaptor, symtab, types)))
      return failure();

    // Create the struct representing the closure back at the CreateClosure site
    rewriter.setInsertionPoint(oldInsertionBlock, oldInsertionPoint);
    if (failed(generateClosureStruct(rewriter, op, adaptor, wrapperFn, types)))
      return failure();

    // Update the subprogram scopes within the wrapper function.
    if (auto sp =
            DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(wrapperFn)) {
      auto newSp = DebugInfo::DISubprogramAttr::get(
          sp.getContext(), sp.getCompileUnit(), sp.getScope(), name, name,
          sp.getFile(), sp.getLine(), sp.getScopeLine(),
          sp.getSubprogramFlags(), sp.getType());
      DebugInfo::DIAttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](DebugInfo::DISubprogramAttr attr) { return newSp; });
      replacer.recursivelyReplaceElementsIn(wrapperFn);
    }

    return success();
  }
};

struct CallSignatureOpConversion
    : public ConvertPOPToLLVMPattern<CallSignatureOp> {
  using ConvertPOPToLLVMPattern<CallSignatureOp>::ConvertPOPToLLVMPattern;
  LogicalResult
  matchAndRewrite(CallSignatureOp op, CallSignatureOpAdaptor adaptor,
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
    auto isClosureType = [](Type type) {
      if (auto sigType = dyn_cast<SignatureType>(type))
        return sigType.isCapturing();
      return false;
    };

    if (isClosureType(callee.getType())) {
      // Unpack the struct representation of the closure.
      auto pointerType = LLVM::LLVMPointerType::get(getContext());
      Value wrapperFnPtr = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 0);
      Value envStruct = rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), adaptor.getCallee(), 1);

      // Compute the type of the wrapper function -- wrapper function type is
      // (!llvm.ptr, unboundArgTy0, ... unboundArgTyn) -> resultTypes
      SmallVector<Type> wrapperFnArgTypes;
      wrapperFnArgTypes.push_back(pointerType);

      auto calleeFuncTy =
          callee.getType().dyn_cast<SignatureType>().getValues();
      for (Type argTy : calleeFuncTy.getInputs()) {
        Type ty = convertType(argTy);
        if (!ty)
          return emitError(op.getLoc())
                 << "could not convert argument type " << argTy;
        wrapperFnArgTypes.push_back(ty);
      }

      auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                       wrapperFnArgTypes, 0);
      Value castWrapperFn = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(wrapperFnType), wrapperFnPtr);

      // Create the call to the wrapper function.
      SmallVector<Value> llvmCallArgs;
      llvmCallArgs.push_back(castWrapperFn);
      llvmCallArgs.push_back(envStruct);
      for (Value inp : adaptor.getArguments())
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
};

void LowerRuntimeClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<KGENDialect>();
  target.addLegalDialect<POP::POPDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  target.addIllegalOp<CreateClosureOp>();
  target.addIllegalOp<CallSignatureOp>();
  patterns.insert<CreateRuntimeClosureOpConversion>(symtab, typeConverter);
  patterns.insert<CallSignatureOpConversion>(typeConverter);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
