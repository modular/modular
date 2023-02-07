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
#include "Support/HLCFDialect/HLCFOps.h"
#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertSymbolOpToLLVM
//===----------------------------------------------------------------------===//

/// This pattern is used to rewrite symbol operations while keeping the symbol
/// table up-to-date.
template <typename OpT>
class ConvertSymbolOpToLLVM : public ConvertPOPToLLVMPattern<OpT> {
public:
  ConvertSymbolOpToLLVM(mlir::LLVMTypeConverter &typeConverter,
                        SymbolTable &symtab)
      : ConvertPOPToLLVMPattern<OpT>(typeConverter), symtab(symtab) {}

protected:
  /// The symbol table.
  SymbolTable &symtab;
};

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
    symtab.remove(func);
    rewriter.eraseOp(func);
    Block::iterator insertPt(func->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENExternFunc
//===----------------------------------------------------------------------===//

/// Convert `kgen.extern.func` to an extern `llvm.func`.
struct ConvertKGENExternFunc : public ConvertSymbolOpToLLVM<ExternFuncOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

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
    symtab.remove(op);
    auto funcOp = rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, op.getNameAttr(), funcType, LLVM::Linkage::External);
    Block::iterator insertPt(funcOp->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENExternVariable
//===----------------------------------------------------------------------===//

/// Convert `kgen.extern.variable` to an extern global variable.
struct ConvertKGENExternVariable
    : public ConvertSymbolOpToLLVM<ExternVariableOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(ExternVariableOp op,
                  typename ExternVariableOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the type of the variable.
    Type llvmType = this->getTypeConverter()->convertType(op.getType());
    if (!llvmType)
      return emitError(op.getLoc(), "failed to convert variable type");

    // Replace it with an LLVM global variable.
    symtab.remove(op);
    auto globalOp = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        op, llvmType, false, LLVM::Linkage::External, op.getName(),
        /*value=*/nullptr);
    Block::iterator insertPt(globalOp->getNextNode());
    globalOp->remove();
    symtab.insert(globalOp, insertPt);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENAddressOf
//===----------------------------------------------------------------------===//

struct ConvertKGENAddressOf : public ConvertPOPToLLVMPattern<AddressOfOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddressOfOp op, AddressOfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type funcPtrType = getTypeConverter()->convertType(op.getType());
    if (!funcPtrType)
      return op.emitError("failed to convert function type");
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, funcPtrType, cast<FlatSymbolRefAttr>(op.getCalleeSymbol()));
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
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), types, flatSymbol, adaptor.getOperands());

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
    return HLCF::lowerReturnOperationToLLVM(op, adaptor.getOperands(), rewriter,
                                            *getTypeConverter());
  }
};

//===----------------------------------------------------------------------===//
// ConvertHLCFReturn
//===----------------------------------------------------------------------===//

/// Convert `hlcf.return` here as well to maintain correctness.
struct ConvertHLCFReturn : public ConvertPOPToLLVMPattern<HLCF::ReturnOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(HLCF::ReturnOp op, HLCF::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return HLCF::lowerReturnOperationToLLVM(op, adaptor.getOperands(), rewriter,
                                            *getTypeConverter());
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

struct ConvertKGENParamConstant
    : public ConvertPOPToLLVMPattern<ParamConstantOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value value = convertParameterToLLVM(b, *getTypeConverter(), op.getValue());
    if (!value)
      return failure();
    rewriter.replaceOp(op, value);
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
                                       mlir::RewritePatternSet &patterns,
                                       SymbolTable &symtab) {
  patterns.insert<
      // clang-format off
      ConvertKGENAddressOf,
      ConvertKGENCall,
      ConvertKGENParamConstant,
      ConvertKGENReturn,
      ConvertHLCFReturn
      // clang-format on
      >(typeConverter);
  patterns.insert<
      // clang-format off
      ConvertKGENFunc,
      ConvertKGENExternFunc,
      ConvertKGENExternVariable
      // clang-format on
      >(typeConverter, symtab);
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

/// Emit a wrapper for a function with the calling convention converted to C
/// calling convention. The wrapper constructs the necessary structs and
/// forwards them to the actual function.
/// The wrapper name is assumed to be unique.
static void emitCWrapper(LLVM::LLVMFuncOp func, StringAttr wrapperName,
                         SymbolTable &symtab) {
  // Generate a new subprogram scope if necessary.
  assert(symtab.lookup(wrapperName) == nullptr && "wrapperName is not unique");
  Location loc = func.getLoc();
  if (auto funcSp = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(loc)) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](DebugInfo::DISubprogramAttr sp) {
      if (sp != funcSp)
        return sp;
      // The symbol name corresponds to the linkage name.
      return DebugInfo::DISubprogramAttr::get(
          sp.getCompileUnit(), sp.getScope(), sp.getName(), wrapperName,
          sp.getFile(), sp.getLine(), sp.getScopeLine(),
          sp.getSubprogramFlags(), sp.getType());
    });
    loc = cast<Location>(replacer.replace(loc));
  }

  // Create the wrapper body. Ownership of the block is handed to the function.
  auto *body = new Block;

  // Convert the calling convention.
  SmallVector<Value> newArgs;
  Type resultType = func.getResultTypes().front();
  ArrayRef<BlockArgument> results = convertCallingConvention(
      loc, body, func.getArguments(), resultType, newArgs);

  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToEnd(body);
  auto call = b.create<LLVM::CallOp>(func, newArgs);

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
  auto wrapper = b.create<LLVM::LLVMFuncOp>(
      wrapperName, LLVM::LLVMFunctionType::get(
                       resultType, llvm::to_vector(body->getArgumentTypes())));
  wrapper.getBody().push_back(body);
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
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  // Capture all the public symbols declared by kgen.export declarations.
  DenseMap<StringAttr, StringAttr> publicSymbols =
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
  theModule->setAttrs(moduleAttrs.getDictionary(&getContext()));

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  SymbolTable symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  populateKGENToLLVMPatterns(typeConverter, patterns, symtab);
  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Set the linkage of symbols marked as public to external.
  for (auto [sym, alias] : publicSymbols) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(sym);
    func.setLinkage(LLVM::Linkage::External);
    // And emit a C wrapper for it.
    emitCWrapper(func, alias, symtab);
  }

  // Convert the debug info within the IR.
  POPToLLVMDebugInfoTypeConverter debugTypeConverter(typeConverter);
  debugTypeConverter.applyRecursively(theModule);
}
