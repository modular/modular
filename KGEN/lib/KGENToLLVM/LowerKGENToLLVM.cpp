//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertKGENFunc
//===----------------------------------------------------------------------===//

struct ConvertKGENFunc : public ConvertSymbolOpToLLVM<FuncOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(FuncOp func, FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(func.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        func.getFunctionType(),
        /*isVariadic=*/false,
        getTypeConverter()->getOptions().useBarePtrCallConv, result);
    if (!funcType)
      return emitError(func.getLoc(), "failed to convert func signature");

    // Mark all functions as internal for now - we'll clean this up later.
    auto funcOp =
        createLLVMFunc(rewriter, getTypeConverter()->getTarget(), func.getLoc(),
                       func.getNameAttr(), funcType, LLVM::Linkage::Internal);

    // And move the func's body into the new function.
    rewriter.inlineRegionBefore(func.getBodyRegion(), funcOp.getBody(),
                                funcOp.end());
    (void)rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());

    // Remove the function.
    symtab.remove(func);
    Block::iterator insertPt(func->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);
    rewriter.eraseOp(func);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENExternFunc
//===----------------------------------------------------------------------===//

struct ConvertKGENExternFunc : public ConvertSymbolOpToLLVM<ExternFuncOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(ExternFuncOp func, ExternFuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the func signature.
    FunctionType signature = func.getFunctionType();
    TypeConverter::SignatureConversion result(signature.getNumInputs());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        signature,
        /*isVariadic=*/false,
        getTypeConverter()->getOptions().useBarePtrCallConv, result);
    if (!funcType)
      return emitError(func.getLoc(), "failed to convert func signature");

    auto funcOp =
        createLLVMFunc(rewriter, getTypeConverter()->getTarget(), func.getLoc(),
                       func.getNameAttr(), funcType, LLVM::Linkage::ExternWeak);

    // Remove the function.
    symtab.remove(func);
    Block::iterator insertPt(func->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);
    rewriter.eraseOp(func);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENCall
//===----------------------------------------------------------------------===//

/// Convert `kgen.call` to `llvm.call`, unpacking results if necessary.
struct ConvertKGENCall : public ConvertPOPToLLVMPattern<CallOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallOp op, CallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result types.
    SmallVector<Type> types = llvm::to_vector(op.getResultTypes());
    if (!types.empty()) {
      types.assign({getTypeConverter()->packFunctionResults(types)});
      if (!types.back())
        return emitError(op.getLoc(), "failed to convert call result type");
    }

    auto flatSymbol = dyn_cast<FlatSymbolRefAttr>(op.getCalleeSymbol());
    if (!flatSymbol)
      return emitError(op.getLoc(),
                       "cannot lower call to nested symbol to LLVM");

    // Create the LLVM call operation.
    LLVM::CallOp llvmCall = createLLVMCall(rewriter, op.getLoc(), types,
                                           flatSymbol, adaptor.getOperands());

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    if (op.getNumResults() <= 1) {
      llvm::append_range(results, llvmCall.getResults());
    } else {
      results.reserve(op.getNumResults());
      for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            op.getLoc(), llvmCall.getResult(), i));
      }
    }

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENReturn
//===----------------------------------------------------------------------===//

/// Convert `kgen.return` to `llvm.return`, packing the results if necessary.
struct ConvertKGENReturn : public ConvertPOPToLLVMPattern<ReturnOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    // If the results don't need to be packed, create the LLVM return.
    if (op->getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), operands);
      return success();
    }

    // Pack the function results in a struct.
    Type type = getTypeConverter()->packFunctionResults(op->getOperandTypes());
    if (!type)
      return emitError(op->getLoc(), "failed to convert return types");
    Value result = rewriter.create<LLVM::UndefOp>(op->getLoc(), type);
    for (auto [index, operand] : llvm::enumerate(operands)) {
      result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result,
                                                    operand, index);
    }

    // Create the LLVM return.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENUnreachable
//===----------------------------------------------------------------------===//

/// Convert `kgen.unreachable` to `llvm.unreachable`.
struct ConvertKGENUnreachable : public ConvertPOPToLLVMPattern<UnreachableOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnreachableOp op, UnreachableOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the llvm.trap + llvm.unreachable ops.
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    rewriter.create<LLVM::CallIntrinsicOp>(op.getLoc(), voidTy, "llvm.trap",
                                           ValueRange());
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamConstant
//===----------------------------------------------------------------------===//

class ConvertKGENParamConstant
    : public ConvertPOPToLLVMPattern<ParamConstantOp> {
public:
  ConvertKGENParamConstant(mlir::LLVMTypeConverter &tc,
                           InterpreterMemoryConverter &imc)
      : ConvertPOPToLLVMPattern(tc), imc(imc) {}

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    InterpreterMemoryConverter::MaterializationScope scope = imc.createScope();
    Value value = convertParameterToLLVM(b, *getTypeConverter(), &imc, &scope,
                                         op.getValue());
    if (!value)
      return failure();
    rewriter.replaceOp(op, value);
    return success();
  }

private:
  /// Convert for global memory references.
  InterpreterMemoryConverter &imc;
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamMaterialize
//===----------------------------------------------------------------------===//

class ConvertKGENParamMaterialize
    : public ConvertPOPToLLVMPattern<ParamMaterializeOp> {
public:
  ConvertKGENParamMaterialize(mlir::LLVMTypeConverter &tc,
                              InterpreterMemoryConverter &imc)
      : ConvertPOPToLLVMPattern(tc), imc(imc) {}

  LogicalResult
  matchAndRewrite(ParamMaterializeOp op, ParamMaterializeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    InterpreterMemoryConverter::MaterializationScope scope = imc.createScope();
    Value value = convertParameterToLLVM(b, *getTypeConverter(), &imc, &scope,
                                         op.getValue());
    if (!value)
      return failure();
    rewriter.replaceOp(op, value);
    return success();
  }

private:
  /// Convert for interpreter memory references.
  InterpreterMemoryConverter &imc;
};

//===----------------------------------------------------------------------===//
// ConvertKGENUndef
//===----------------------------------------------------------------------===//

struct ConvertKGENUndef : public ConvertPOPToLLVMPattern<UndefOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(UndefOp op, UndefOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return emitError(op->getLoc(), "failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, type);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENGlobalAddress
//===----------------------------------------------------------------------===//

struct ConvertKGENGlobalAddress
    : public ConvertPOPToLLVMPattern<GlobalAddressOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(GlobalAddressOp op,
                                GlobalAddressOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Type type = convertType(op.getType());
    if (!type)
      return b.notifyMatchFailure(op.getLoc(), "failed to convert result type");
    // Trivial lowering to `llvm.mlir.addressof`.
    b.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, type, cast<FlatSymbolRefAttr>(op.getGlobal()));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static LogicalResult removeLinkOps(LinkOp linkOp, PatternRewriter &rewriter) {
  rewriter.eraseOp(linkOp);
  return success();
}

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       SymbolTable &symtab,
                                       InterpreterMemoryConverter &imc) {
  patterns.insert<
      // clang-format off
      ConvertKGENCall,
      ConvertKGENGlobalAddress,
      ConvertKGENReturn,
      ConvertKGENUnreachable,
      ConvertKGENUndef
      // clang-format on
      >(typeConverter);
  patterns.insert<
      // clang-format off
      ConvertKGENFunc,
      ConvertKGENExternFunc
      // clang-format on
      >(typeConverter, symtab);
  patterns.insert<ConvertKGENParamConstant, ConvertKGENParamMaterialize>(
      typeConverter, imc);
  // Just remove LinkOps.
  patterns.add(removeLinkOps);
}

//===----------------------------------------------------------------------===//
// convertGlobals
//===----------------------------------------------------------------------===//

static LogicalResult convertGlobals(ModuleOp module, POPToLLVMTypeConverter &tc,
                                    bool disableGlobalDtors) {
  SmallVector<Attribute> ctors, dtors, priorities;

  for (auto global : llvm::make_early_inc_range(module.getOps<GlobalOp>())) {
    // Replace the `pop.global` with an `llvm.mlir.global`, raise the
    // constructor and destructor into functions, and collect a list of them.
    mlir::IRRewriter b{OpBuilder(global)};
    Type type = tc.convertType(global.getType());
    if (!type)
      return global.emitError("could not convert global type");

    if (global.getCtor()) {
      ctors.push_back(*global.getCtor());
      dtors.push_back(*global.getDtor());
      priorities.push_back(global.getPriorityAttr());
    }

    // Create the LLVM global.
    bool isExported = global.isExported();
    auto llvmGlobal = b.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, type, /*constant=*/false,
        isExported ? LLVM::Linkage::External : LLVM::Linkage::Internal,
        global.getSymName(), /*value=*/Attribute());

    // If the global is not exported, then no need to initialize it.
    if (!isExported)
      continue;

    // If the global is exported, explicitly initialize it as undef.
    b.createBlock(&llvmGlobal.getBodyRegion());
    Value undef = b.create<LLVM::UndefOp>(llvmGlobal.getLoc(), type);
    b.create<LLVM::ReturnOp>(llvmGlobal.getLoc(), undef);
  }

  // Don't generate anything if there are no globals.
  if (ctors.empty())
    return success();

  // Create the `llvm.mlir.global_ctors` and `llvm.mlir.global_dtors`.
  auto b = OpBuilder::atBlockBegin(module.getBody());
  mlir::ArrayAttr prioritiesAttr = b.getArrayAttr(priorities);
  b.create<LLVM::GlobalCtorsOp>(module.getLoc(), b.getArrayAttr(ctors),
                                prioritiesAttr);
  // FIXME(#16605): Global destructors don't work in JIT mode.
  if (!disableGlobalDtors)
    b.create<LLVM::GlobalDtorsOp>(module.getLoc(), b.getArrayAttr(dtors),
                                  prioritiesAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// Emit C API Wrappers
//===----------------------------------------------------------------------===//

/// Convert the calling convention of the argument type.
static Value convertArgCallingConvention(ImplicitLocOpBuilder &b, Type type,
                                         Block *body) {
  // Recursively flatten a struct type into the function argument list. Pack
  // the struct from the flat arguments and return it.
  auto flattenArgumentStruct = [&](LLVM::LLVMStructType structTy) {
    Value result = b.create<LLVM::UndefOp>(structTy);
    for (auto [index, type] : llvm::enumerate(structTy.getBody())) {
      Value value = convertArgCallingConvention(b, type, body);
      result = b.create<LLVM::InsertValueOp>(result, value, index);
    }
    return result;
  };

  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type))
    return flattenArgumentStruct(structTy);
  if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(type)) {
    // Change the array to be pass-by-reference.
    Value arrPtr =
        body->addArgument(LLVM::LLVMPointerType::get(arrayTy), b.getLoc());
    return b.create<LLVM::LoadOp>(arrPtr);
  }
  return body->addArgument(type, b.getLoc());
}

/// Recursively flatten a result struct type into the argument list.
static unsigned flattenResultStruct(Location loc, LLVM::LLVMStructType structTy,
                                    Block *body) {
  unsigned numAdded = 0;
  for (Type type : structTy.getBody()) {
    if (auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(type)) {
      numAdded += flattenResultStruct(loc, nestedStruct, body);
    } else {
      body->addArgument(LLVM::LLVMPointerType::get(type), loc);
      ++numAdded;
    }
  }
  return numAdded;
}

/// Recursively unpack the struct and store the nested values into pointer
/// arguments.
static void flattenResultStruct(ImplicitLocOpBuilder &b,
                                LLVM::LLVMStructType structTy, Value result,
                                ArrayRef<BlockArgument> results,
                                unsigned &idx) {
  for (auto [index, type] : llvm::enumerate(structTy.getBody())) {
    Value value = b.create<LLVM::ExtractValueOp>(result, index);
    if (auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(type))
      flattenResultStruct(b, nestedStruct, value, results, idx);
    else
      b.create<LLVM::StoreOp>(value, results[idx++]);
  }
}

/// Rewrite the given arguments to be compatible with C calling conventions.
/// Break up the structs in the given arguments and result type and rewrite
/// arrays to be pass-by-reference. Append new arguments to `body` and populate
/// `newArgs` with the packed structs created at the top of the body.
static void convertArgCallingConvention(Location loc, Block *body,
                                        ArrayRef<BlockArgument> args,
                                        SmallVectorImpl<Value> &newArgs) {
  // Flatten structs in the argument list.
  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToStart(body);
  for (Value arg : args) {
    b.setLoc(arg.getLoc());
    newArgs.push_back(convertArgCallingConvention(b, arg.getType(), body));
  }
}

/// Rewrite the given result type to be compatible with C calling conventions.
/// Break up the structs in the given result type. Append new arguments to
/// `body`, and return the slice of arguments that represent the result
/// arguments.
static ArrayRef<BlockArgument>
convertResultCallingConvention(Location loc, Block *body, Type resultTy) {
  // Flatten the results if necessary at all the return points.
  ArrayRef<BlockArgument> results;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    unsigned numAdded = flattenResultStruct(loc, structTy, body);
    results = body->getArguments().take_back(numAdded);
  }
  return results;
}

/// Emit a wrapper for a function with the calling convention converted to C
/// calling convention. The wrapper constructs the necessary structs and
/// forwards them to the actual function.
static void emitCWrapper(LLVM::LLVMFuncOp func,
                         mlir::SymbolUserMap &symbolUsers, SymbolTable &symtab,
                         TargetInfoAttr target) {
  // The function has internal users. Update its symbol name so the wrapper can
  // take its name.
  StringAttr origName = func.getSymNameAttr();
  auto newName = StringAttr::get(
      func.getContext(),
      getUniqueSymbolName((origName.getValue() + "_c_wrapped").str(), symtab));
  symbolUsers.replaceAllUsesWith(func, newName);
  symtab.remove(func);
  func.setSymNameAttr(newName);
  symtab.insert(func);

  // Update the subprogram scope of the wrapped function if it has one, but save
  // the location before it gets changed.
  Location loc = func.getLoc();
  DebugInfo::updateSubprogram(func, newName, newName);

  // Create the wrapper body. Ownership of the block is handed to the function.
  auto *body = new Block;

  // Convert the calling convention.
  SmallVector<Value> newArgs;
  convertArgCallingConvention(loc, body, func.getArguments(), newArgs);
  Type resultType = func.getFunctionType().getReturnType();
  ArrayRef<BlockArgument> results =
      convertResultCallingConvention(loc, body, resultType);

  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToEnd(body);
  LLVM::CallOp call = createLLVMCall(b, b.getLoc(), func, newArgs);

  // If the result type is a struct, flatten it into the arguments.
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultType)) {
    resultType = LLVM::LLVMVoidType::get(func.getContext());
    unsigned idx = 0;
    flattenResultStruct(b, structTy, call.getResult(), results, idx);
    b.create<LLVM::ReturnOp>(ValueRange());
  } else {
    b.create<LLVM::ReturnOp>(call.getResults());
  }

  b.setInsertionPointAfter(func);
  auto wrapper = createLLVMFunc(
      b, target, b.getLoc(), origName,
      LLVM::LLVMFunctionType::get(resultType,
                                  llvm::to_vector(body->getArgumentTypes())));
  wrapper.getBody().push_back(body);
}

/// Process the given function which is exported to C. If possible this will try
/// to update the function in place, otherwise a wrapper is emitted that
/// internally invokes the provided function.
static void processCExportedFunction(LLVM::LLVMFuncOp func,
                                     mlir::SymbolUserMap &symbolUsers,
                                     SymbolTable &symtab,
                                     TargetInfoAttr target) {
  // Check if we need to update the function arguments or results to be
  // C-compatible.
  ArrayRef<Type> currentFunctionTypes = func.getArgumentTypes();
  Type resultType = func.getFunctionType().getReturnType();
  bool needUpdatedArgTypes = llvm::any_of(currentFunctionTypes, [](Type type) {
    return isa<LLVM::LLVMArrayType, LLVM::LLVMStructType>(type);
  });
  bool needUpdatedResultType = isa<LLVM::LLVMStructType>(resultType);

  // If we need to update the calling convention and we have internal users,
  // emit a wrapper function as the structure of the function will have to
  // change.
  bool hasInternalUsers = !symbolUsers.getUsers(func).empty();
  if ((needUpdatedArgTypes || needUpdatedResultType) && hasInternalUsers)
    return emitCWrapper(func, symbolUsers, symtab, target);

  // Otherwise, we can update the function in place.
  func.setLinkage(LLVM::Linkage::External);

  // If we don't need to update the calling convention, we're done.
  if (!needUpdatedArgTypes && !needUpdatedResultType)
    return;
  Block *entryBlock = &func.getBody().front();

  // Check to see if we need to update any of the function arguments.
  if (needUpdatedArgTypes) {
    SmallVector<Value> newArgs;
    convertArgCallingConvention(func.getLoc(), entryBlock,
                                llvm::to_vector(func.getArguments()), newArgs);

    // Replace the original arguments with the new ones.
    for (unsigned i = 0, e = newArgs.size(); i != e; ++i)
      func.getArgument(i).replaceAllUsesWith(newArgs[i]);
    entryBlock->eraseArguments(0, currentFunctionTypes.size());
  }

  // Check if the result type needs updating.
  if (needUpdatedResultType) {
    ArrayRef<BlockArgument> results =
        convertResultCallingConvention(func.getLoc(), entryBlock, resultType);

    // Replace the original results with the new ones.
    auto structTy = cast<LLVM::LLVMStructType>(resultType);
    resultType = LLVM::LLVMVoidType::get(func.getContext());

    // Update all of the returns within the function.
    func.walk([&](LLVM::ReturnOp returnOp) {
      unsigned idx = 0;
      ImplicitLocOpBuilder b(returnOp.getLoc(), returnOp);
      flattenResultStruct(b, structTy, returnOp.getArg(), results, idx);
      returnOp->setOperands(ValueRange());
    });
  }

  // Update the function type.
  func.setType(LLVM::LLVMFunctionType::get(
      resultType, llvm::to_vector(entryBlock->getArgumentTypes())));
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENToLLVMPass
    : public KGEN::impl::LowerKGENToLLVMBase<LowerKGENToLLVMPass> {
  using LowerKGENToLLVMBase::LowerKGENToLLVMBase;

  void runOnOperation() override;
};
} // namespace

void LowerKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<KGENDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<POP::POPDialect>();
  target.addLegalDialect<mlir::index::IndexDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addLegalOp<KGEN::CallSignatureOp>();
  target.addLegalOp<KGEN::CreateClosureOp>();

  // Capture all the exported symbols.
  llvm::MapVector<StringAttr, ExportedSymbol> publicSymbols =
      getExportedSymbols(theModule);

  // Configure the type converter.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Attach the LLVM data layout and target triple strings to the module so they
  // are present when exporting to LLVMIR.
  NamedAttrList moduleAttrs(theModule->getAttrDictionary());
  moduleAttrs.set(LLVM::LLVMDialect::getTargetTripleAttrName(),
                  StringAttr::get(&getContext(), targetInfo.getTripleStr()));
  moduleAttrs.set(
      LLVM::LLVMDialect::getDataLayoutAttrName(),
      StringAttr::get(&getContext(), targetInfo.getDataLayout().toString()));
  moduleAttrs.erase(EnvAttr::getEnvAttrName());
  theModule->setAttrs(moduleAttrs.getDictionary(&getContext()));

  // Convert global ops and generator global constructors and destructors.
  if (failed(convertGlobals(theModule, typeConverter, disableGlobalDtors)))
    return signalPassFailure();

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable &symtab = symtabAnalysis.getTopLevelSymbolTable();
  InterpreterMemoryConverter imc(symtab, typeConverter);
  populateKGENToLLVMPatterns(typeConverter, patterns, symtab, imc);
  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Process updates to any exported functions.
  mlir::SymbolUserMap symbolUsers(symtabAnalysis.getSymbolTables(), theModule);
  for (auto [sym, exportSymbol] : publicSymbols) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(sym);
    // If the function is not C exported, just update its linkage. Otherwise,
    // generate a wrapper function.
    if (!func)
      continue;
    else if (!exportSymbol.isCExport)
      func.setLinkage(func.isExternal() ? LLVM::Linkage::External
                                        : LLVM::Linkage::Weak);
    else
      processCExportedFunction(func, symbolUsers, symtab, targetInfo);
  }

  // Convert the debug info within the IR.
  POPToLLVMDebugInfoTypeConverter debugTypeConverter(typeConverter, targetInfo);
  debugTypeConverter.applyRecursively(theModule);
}
