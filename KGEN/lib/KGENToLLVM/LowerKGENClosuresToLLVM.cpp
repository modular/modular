//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLVMLoweringUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/BinaryFormat/Dwarf.h"

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
  using LowerRuntimeClosuresBase::LowerRuntimeClosuresBase;

  void runOnOperation() override;
};
} // namespace

struct CreateClosureTypes {
public:
  CreateClosureTypes() {}
  static LogicalResult
  createClosureTypes(CreateClosureTypes &types, CreateClosureOp op,
                     const mlir::LLVMTypeConverter &typeConverter);
  SmallVector<Type> boundArgTypes;
  Type opaquePtrType;
  /// Two element struct where first element is void* to function
  /// and second member is struct of captured elements.
  Type liftedFunctionCaptureType;

  /// The opaque closure struct type is a struct of two opaque pointers
  /// The first element points to a function interface method.
  /// The second points to captured state + lifted function.
  Type unpackingFunctionAndCapturesType;
};

LogicalResult CreateClosureTypes::createClosureTypes(
    CreateClosureTypes &types, CreateClosureOp op,
    const mlir::LLVMTypeConverter &typeConverter) {
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
  unsigned nameIndex = 0;

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
        context, packedResTy, wrapperFnArgTypes, /*varArg=*/false);

    // Create the body of the wrapper function
    Block *wrapperFnBody = new Block;
    rewriter.setInsertionPointToStart(wrapperFnBody);

    // Add wrapper function arguments to the block arguments
    for (Type argTy : wrapperFnArgTypes)
      wrapperFnBody->addArgument(argTy, op.getLoc());
    rewriter.clearInsertionPoint();

    LLVM::LLVMFuncOp wrapperFn = createLLVMFunc(
        rewriter, getTypeConverter()->getTarget(), op.getLoc(),
        rewriter.getStringAttr(
            "closure_wrapper_fn_" +
            Twine(const_cast<CreateRuntimeClosureOpConversion *>(this)
                      ->nameIndex++)),
        wrapperFnType, LLVM::Linkage::Internal);

    // If possible, we need to add a subprogram scope to the new function.
    auto scope = DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
        op.getLoc(), DebugInfo::LocWalkPolicy::CalleePriority);
    if (scope) {
      // Use unresolved types now for simplicity, these will get resolved during
      // compilation.
      auto mapUnresolvedType = [&](Type type) -> DebugInfo::DIType {
        return DebugInfo::DIUnresolvedMLIRType::get(type);
      };
      auto spType = DebugInfo::DISubroutineType::get(
          op->getContext(), map_to_vector(wrapperFnArgTypes, mapUnresolvedType),
          map_to_vector(wrapperFnType.getReturnTypes(), mapUnresolvedType));

      auto fileLoc = op.getLoc()->findInstanceOf<FileLineColLoc>();
      auto sourceName = DebugInfo::SourceNameAttr::get(
          "closure_wrapper_fn." + Twine(nameIndex - 1), scope.getName());
      wrapperFn->setLoc(FusedLoc::get(
          op.getContext(), Location(fileLoc),
          scope.cloneWith(sourceName, wrapperFn.getSymNameAttr(), spType)));
    }

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
    auto func =
        symbolTable.lookup<LLVM::LLVMFuncOp>(flatSymbol.getRootReference());
    if (!func)
      return emitError(op.getLoc(), "Callee does not reference llvm function");

    for (size_t i = 0; i < op.getCaptures().size(); i++) {
      Type capturedArgType = types.boundArgTypes[i];
      LLVM::GEPOp boundArgPtr = rewriter.create<LLVM::GEPOp>(
          wrapperFn.getLoc(), types.opaquePtrType, types.opaquePtrType,
          wrapperFnBody.getArgument(0),
          ArrayRef<LLVM::GEPArg>({0, static_cast<int32_t>(i)}));
      boundArgPtr.setElemType(types.liftedFunctionCaptureType);
      Value boundArg = rewriter.create<LLVM::LoadOp>(
          wrapperFn.getLoc(), capturedArgType, boundArgPtr);
      liftedNestedFunctionCallArgs[i] = boundArg;
    }
    size_t numCaptures = op.getCaptures().size();
    size_t numberDynamicArgs =
        op.getCalleeType().getValues().getNumInputs() - numCaptures;
    for (size_t i = 0; i < numberDynamicArgs; i++)
      liftedNestedFunctionCallArgs[i + numCaptures] =
          wrapperFnBody.getArgument(i + 1);

    ValueRange valueRange(liftedNestedFunctionCallArgs);
    LLVM::CallOp callLiftedFunction =
        createLLVMCall(rewriter, wrapperFn.getLoc(), func, valueRange);
    rewriter.create<LLVM::ReturnOp>(wrapperFn.getLoc(),
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
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, addressOfWrapperFunction, 0);
    Value one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(context, 8), 1);
    LLVM::AllocaOp envStruct =
        rewriter.create<LLVM::AllocaOp>(op.getLoc(), types.opaquePtrType, one);
    envStruct.setElemType(types.liftedFunctionCaptureType);
    // TODO: When data layouts are propagated properly, extract the data
    //  layout from TargetInfoAttr
    size_t envSize =
        getTypeConverter()->getTypeAllocSize(types.liftedFunctionCaptureType);

    rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), envSize, envStruct);
    for (auto [argIdx, boundArgValue] :
         llvm::enumerate(adaptor.getCaptures())) {
      LLVM::GEPOp getBoundArgPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), /*resultType=*/types.opaquePtrType,
          /*basePtrType=*/types.opaquePtrType, /*basePtr=*/envStruct,
          ArrayRef<LLVM::GEPArg>({0, static_cast<int32_t>(argIdx)}));
      getBoundArgPtr.setElemType(types.liftedFunctionCaptureType);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), boundArgValue,
                                     getBoundArgPtr.getResult());
    }

    // Add the environment struct to the closure struct
    closureStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), closureStruct, envStruct, 1);

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
    symtab.insert(wrapperFn);
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

    return success();
  }
};

struct CallIndirectOpConversion
    : public ConvertPOPToLLVMPattern<CallIndirectOp> {
  using ConvertPOPToLLVMPattern<CallIndirectOp>::ConvertPOPToLLVMPattern;
  LogicalResult
  matchAndRewrite(CallIndirectOp op, CallIndirectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If there are no result types, set it to void. Otherwise, set the result
    // type to the packed result types.
    Type resultType;
    if (!op.getNumResults())
      resultType = getVoidType();
    else
      resultType = getTypeConverter()->packFunctionResults(op.getResultTypes());
    if (!resultType)
      return emitError(op.getLoc(), "failed to convert call result type");

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

      auto calleeFuncTy = cast<SignatureType>(callee.getType()).getValues();
      for (Type argTy : calleeFuncTy.getInputs()) {
        Type ty = convertType(argTy);
        if (!ty)
          return emitError(op.getLoc())
                 << "could not convert argument type " << argTy;
        wrapperFnArgTypes.push_back(ty);
      }

      auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                       wrapperFnArgTypes, 0);

      // Create the call to the wrapper function.
      SmallVector<Value> llvmCallArgs;
      llvmCallArgs.push_back(wrapperFnPtr);
      llvmCallArgs.push_back(envStruct);
      llvm::append_range(llvmCallArgs, adaptor.getArguments());

      llvmCall =
          createLLVMCall(rewriter, op.getLoc(), wrapperFnType, llvmCallArgs);
      if (op.getTailKind().has_value()) {
        switch (op.getTailKind().value()) {
        case TailKind::MustTail:
          llvmCall.setTailCallKind(LLVM::tailcallkind::TailCallKind::MustTail);
          break;
        case TailKind::NoTail:
          llvmCall.setTailCallKind(LLVM::tailcallkind::TailCallKind::MustTail);
          break;
        default:
          break;
        }
      }
    } else {
      // Create the LLVM call operation.
      // Note: adaptor.getOperands() is a list of callee followed by inputs.
      SmallVector<Type> wrapperFnArgTypes;
      llvm::append_range(wrapperFnArgTypes, adaptor.getArguments().getTypes());
      auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                       wrapperFnArgTypes, 0);
      llvmCall = createLLVMCall(rewriter, op.getLoc(), wrapperFnType,
                                adaptor.getOperands());
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
  target.addIllegalOp<CallIndirectOp>();
  patterns.insert<CreateRuntimeClosureOpConversion>(symtab, typeConverter);
  patterns.insert<CallIndirectOpConversion>(typeConverter);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
