//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/ML/DType.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertKGENFunc
//===----------------------------------------------------------------------===//

struct ConvertKGENFunc : public mlir::ConvertOpToLLVMPattern<FuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FuncOp func, FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(func.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        func.getFunctionType(),
        /*isVariadic=*/false, result);
    if (!funcType)
      return emitError(func.getLoc(), "failed to convert func signature");

    // Mark all functions as internal for now - we'll clean this up later.
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        func.getLoc(), func.getNameAttr(), funcType, LLVM::Linkage::Internal);

    // And move the func's body into the new function.
    rewriter.inlineRegionBefore(func.getBodyRegion(), funcOp.getBody(),
                                funcOp.end());
    (void)rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());

    // Remove the function.
    rewriter.eraseOp(func);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENExternFunc
//===----------------------------------------------------------------------===//

/// Convert `kgen.extern.func` to an extern `llvm.func`.
struct ConvertKGENExternFunc
    : public mlir::ConvertOpToLLVMPattern<ExternFuncOp> {
  using mlir::ConvertOpToLLVMPattern<ExternFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExternFuncOp op, typename ExternFuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(
        op.getFunctionType().getNumInputs());
    Type funcType = this->getTypeConverter()->convertFunctionSignature(
        op.getFunctionType(),
        /*isVariadic=*/false, result);
    if (!funcType)
      return emitError(op.getLoc(), "failed to convert func signature");

    // Replace it with an LLVM function that has no body.
    rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, op.getNameAttr(), funcType, LLVM::Linkage::External);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENExternVariable
//===----------------------------------------------------------------------===//

/// Convert `kgen.extern.variable` to an extern global variable.
struct ConvertKGENExternVariable
    : public mlir::ConvertOpToLLVMPattern<ExternVariableOp> {
  using mlir::ConvertOpToLLVMPattern<ExternVariableOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExternVariableOp op,
                  typename ExternVariableOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the type of the variable.
    Type llvmType = this->getTypeConverter()->convertType(op.getType());
    if (!llvmType)
      return emitError(op.getLoc(), "failed to convert variable type");

    // Replace it with an LLVM global variable.
    rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        op, llvmType, false, LLVM::Linkage::External, op.getName(),
        /*value=*/nullptr);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENAddressOf
//===----------------------------------------------------------------------===//

struct ConvertKGENAddressOf : public mlir::ConvertOpToLLVMPattern<AddressOfOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddressOfOp op, AddressOfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type funcPtrType = getTypeConverter()->convertType(op.getType());
    if (!funcPtrType)
      return op.emitError("failed to convert function type");
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, funcPtrType,
                                                   op.getCalleeAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENCall
//===----------------------------------------------------------------------===//

/// Convert `kgen.call` to `llvm.call`, unpacking results if necessary.
struct ConvertKGENCall : public mlir::ConvertOpToLLVMPattern<CallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

    // Create the LLVM call operation.
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), types, op.getCalleeAttr(), adaptor.getOperands());

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
struct ConvertKGENReturn : public mlir::ConvertOpToLLVMPattern<ReturnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the results don't need to be packed, create the LLVM return.
    if (op.getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(),
                                                  adaptor.getOperands());
      return success();
    }

    // Pack the function results in a struct.
    Type type = getTypeConverter()->packFunctionResults(op.getOperandTypes());
    if (!type)
      return emitError(op.getLoc(), "failed to convert return types");
    Value result = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
    for (auto &it : llvm::enumerate(adaptor.getOperands())) {
      result = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), result,
                                                    it.value(), it.index());
    }

    // Create the LLVM return.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

struct ConvertKGENParamConstant
    : public mlir::ConvertOpToLLVMPattern<ParamConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto dtype = dyn_cast<DTypeConstantAttr>(op.getValue())) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, rewriter.getI8Type(), dtype.getDType().getValue());
    } else if (auto attr = dyn_cast<TypedAttr>(op.getValue());
               attr && isa<IntegerAttr, FloatAttr>(attr)) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, getTypeConverter()->convertType(attr.getType()), attr);
    } else {
      // No support for strings, type constants, or symbol references.
      return op.emitError("unknown parameter value type");
    }
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static LogicalResult removeExportOps(ExportOp exportOp,
                                     PatternRewriter &rewriter) {
  rewriter.eraseOp(exportOp);
  return success();
}

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertKGENAddressOf,
      ConvertKGENCall,
      ConvertKGENFunc,
      ConvertKGENExternFunc,
      ConvertKGENExternVariable,
      ConvertKGENParamConstant,
      ConvertKGENReturn
      // clang-format on
      >(typeConverter);
  // Just remove ExportOps.
  patterns.add(removeExportOps);
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
    for (auto &type : llvm::enumerate(structTy.getBody())) {
      Value value = convertArgCallingConvention(b, type.value(), body);
      result = b.create<LLVM::InsertValueOp>(result, value, type.index());
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
  for (auto &type : llvm::enumerate(structTy.getBody())) {
    Value value = b.create<LLVM::ExtractValueOp>(result, type.index());
    if (auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(type.value()))
      flattenResultStruct(b, nestedStruct, value, results, idx);
    else
      b.create<LLVM::StoreOp>(value, results[idx++]);
  }
}

/// Rewrite the given arguments and result type to be compatible with C calling
/// conventions. Break up the structs in the given arguments and result type and
/// rewrite arrays to be pass-by-reference. Append new arguments to `body` and
/// populate `newArgs` with the packed structs created at the top of the body.
/// Return the slice of arguments that represent the result arguments.
static ArrayRef<BlockArgument>
convertCallingConvention(Location loc, Block *body,
                         ArrayRef<BlockArgument> args, Type resultTy,
                         SmallVectorImpl<Value> &newArgs) {
  // Flatten structs in the argument list.
  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToStart(body);
  for (Value arg : args) {
    b.setLoc(arg.getLoc());
    newArgs.push_back(convertArgCallingConvention(b, arg.getType(), body));
  }

  // Flatten the results if necessary at all the return points.
  ArrayRef<BlockArgument> results;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    unsigned numAdded = flattenResultStruct(loc, structTy, body);
    results = body->getArguments().take_back(numAdded);
  }

  return results;
}

/// Convert the calling convention of the provided function in-place. The
/// function must be top-level as callsites are not modified.
static void rewriteCallingConventionInPlace(LLVM::LLVMFuncOp func) {
  // If there are no argument types to rewrite, return early.
  auto needRewrite = [](Type type) {
    return type.isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>();
  };
  if (!llvm::any_of(func.getArgumentTypes(), needRewrite) &&
      !needRewrite(func.getResultTypes().front()))
    return;

  Block *entry = &func.getBody().front();
  Type resultTy = func.getResultTypes().front();
  SmallVector<Value> newArgs;
  ArrayRef<BlockArgument> results = convertCallingConvention(
      func.getLoc(), entry, llvm::to_vector(func.getArguments()), resultTy,
      newArgs);

  // Flatten the results if necessary at all the return points.
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    resultTy = LLVM::LLVMVoidType::get(func.getContext());
    for (auto ret : llvm::make_early_inc_range(func.getOps<LLVM::ReturnOp>())) {
      ImplicitLocOpBuilder b(ret.getLoc(), ret);
      unsigned idx = 0;
      flattenResultStruct(b, structTy, ret.getOperand(0), results, idx);
      b.create<LLVM::ReturnOp>(ValueRange());
      ret->erase();
    }
  }

  // Replace and erase the old arguments.
  for (auto [arg, newArg] : llvm::zip(
           entry->getArguments().take_front(func.getNumArguments()), newArgs))
    arg.replaceAllUsesWith(newArg);
  entry->eraseArguments(0, func.getNumArguments());

  // Update the function type.
  func.setFunctionTypeAttr(TypeAttr::get(LLVM::LLVMFunctionType::get(
      resultTy, llvm::to_vector(entry->getArgumentTypes()))));
}

/// Break up argument and result structs in-place for the given top-level
/// funcs and emit C wrappers for specific non-top-level funcs.
static LogicalResult emitWrappers(ModuleOp theModule,
                                  ArrayRef<std::string> topLevelFuncs) {
  // Ensure that top-level funcs do not have callsites.
  llvm::StringMap<LLVM::CallOp> callsites;
  for (auto func : theModule.getOps<LLVM::LLVMFuncOp>())
    for (auto call : func.getOps<LLVM::CallOp>())
      if (Optional<StringRef> callee = call.getCallee())
        callsites.try_emplace(*callee, call);

  // Break up structs in-place in the specific top-level funcs.
  SymbolTable symtab(theModule);
  for (StringRef funcName : topLevelFuncs) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(funcName);
    if (!func)
      return theModule.emitError("cannot find func: @") << funcName;
    // If the function's linkage is private, don't bother creating a wrapper.
    if (func.getLinkage() != LLVM::Linkage::External)
      continue;

    if (auto it = callsites.find(funcName); it != callsites.end()) {
      return func.emitError("func is not top-level")
                 .attachNote(it->second.getLoc())
             << "callsite here";
    }
    if (func.isExternal())
      return func.emitError("cannot break up structs of an external function");

    rewriteCallingConventionInPlace(func);
  }
  return success();
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

  explicit LowerKGENToLLVMPass(ArrayRef<StringRef> topLevelFuncs) {
    for (StringRef topLevelFunc : topLevelFuncs)
      this->topLevelFuncs.push_back(topLevelFunc.str());
  }

  void runOnOperation() override;
};
} // namespace

void LowerKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<KGENDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Capture all the public symbols declared by kgen.export declarations.
  SmallVector<FlatSymbolRefAttr> publicSymbols;
  for (auto e : theModule.getOps<ExportOp>())
    for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
      publicSymbols.push_back(sym);

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);

  // Configure the type converter.
  POPToLLVMTypeConverter typeConverter(theModule->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateKGENToLLVMPatterns(typeConverter, patterns);
  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Fix up the linkage for the exported symbols.
  SymbolTable symtab(theModule);
  for (FlatSymbolRefAttr sym : publicSymbols) {
    // Have to add the public symbols to the topLevelFuncs list.
    topLevelFuncs.push_back(sym.getValue().str());

    // And if it's public, set it to external linkage.
    if (auto llvmFunc = symtab.lookup<LLVM::LLVMFuncOp>(sym.getAttr()))
      llvmFunc.setLinkage(LLVM::Linkage::External);
  }

  // Break up structs in top-level funcs exposed to C.
  if (failed(emitWrappers(theModule, topLevelFuncs)))
    return signalPassFailure();

  // Convert the debug info within the IR.
  POPToLLVMDebugInfoTypeConverter debugTypeConverter(typeConverter);
  debugTypeConverter.applyRecursively(theModule);
}
