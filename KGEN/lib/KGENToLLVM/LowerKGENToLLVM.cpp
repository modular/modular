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
        /*isVariadic=*/false,
        getTypeConverter()->getOptions().useBarePtrCallConv, result);
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
        op.getLoc(), types, flatSymbol, adaptor.getOperands(),
        LLVM_FASTMATH_FLAGS,
        /*branch_weights=*/nullptr, /*access_groups=*/nullptr,
        /*alias_scopes*/ nullptr, /*noalias_scopes*/ nullptr, /*tbaa*/ nullptr);

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
// ConvertKGENParamValue
//===----------------------------------------------------------------------===//

struct ConvertKGENParamConstant
    : public ConvertSymbolOpToLLVM<ParamConstantOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(ParamConstantOp op, ParamConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value value =
        convertParameterToLLVM(b, *getTypeConverter(), symtab, op.getValue());
    if (!value)
      return failure();
    rewriter.replaceOp(op, value);
    return success();
  }
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
      ConvertKGENCall,
      ConvertKGENReturn,
      ConvertKGENUnreachable,
      ConvertKGENUndef
      // clang-format on
      >(typeConverter);
  patterns.insert<ConvertKGENParamConstant, ConvertKGENFunc>(typeConverter,
                                                             symtab);
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
  convertArgCallingConvention(loc, body, func.getArguments(), newArgs);
  Type resultType = func.getResultTypes().front();
  ArrayRef<BlockArgument> results =
      convertResultCallingConvention(loc, body, resultType);

  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToEnd(body);
  auto call = b.create<LLVM::CallOp>(func, newArgs);
  call.setFastmathFlags(LLVM_FASTMATH_FLAGS);

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

/// Update the name and linkage of the given function, using the provided alias
/// name.
static void updateExportedFunctionNameAndLinkage(
    LLVM::LLVMFuncOp func, StringAttr aliasName,
    mlir::SymbolUserMap &symbolUsers, SymbolTable &symtab) {
  MLIRContext *ctx = func.getContext();
  NamedAttrList attrs(func->getAttrDictionary());

  // Update the linkage.
  attrs.set(func.getLinkageAttrName(),
            LLVM::LinkageAttr::get(ctx, LLVM::Linkage::External));

  // If the name is the same, there's nothing more to do.
  if (func.getSymNameAttr() == aliasName) {
    func->setAttrs(attrs.getDictionary(ctx));
    return;
  }
  assert(symtab.lookup(aliasName) == nullptr && "aliasName is not unique");

  // Generate a new subprogram scope if necessary with the updated linkage
  // name.
  if (auto funcSp =
          DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(func.getLoc())) {
    DebugInfo::DIAttrTypeReplacer replacer;
    replacer.addReplacement([&](DebugInfo::DISubprogramAttr sp) {
      if (sp != funcSp)
        return sp;
      // The symbol name corresponds to the linkage name.
      return DebugInfo::DISubprogramAttr::get(
          sp.getCompileUnit(), sp.getScope(), sp.getName(), aliasName,
          sp.getFile(), sp.getLine(), sp.getScopeLine(),
          sp.getSubprogramFlags(), sp.getType());
    });
    replacer.recursivelyReplaceElementsIn(func);
  }

  // Update any uses of the function.
  symbolUsers.replaceAllUsesWith(func, aliasName);

  // Update the name within the symbol table.
  symtab.remove(func);
  attrs.set(func.getSymNameAttrName(), aliasName);
  func->setAttrs(attrs.getDictionary(ctx));
  symtab.insert(func);
}

/// Process the given function which is exported to C. If possible this will try
/// to update the function in place, otherwise a wrapper is emitted that
/// internally invokes the provided function.
static void processCExportedFunction(LLVM::LLVMFuncOp func,
                                     StringAttr aliasName,
                                     mlir::SymbolUserMap &symbolUsers,
                                     SymbolTable &symtab) {
  // Check if we need to update the function arguments or results to be
  // C-compatible.
  ArrayRef<Type> currentFunctionTypes = func.getArgumentTypes();
  Type resultType = func.getResultTypes().front();
  bool needUpdatedArgTypes = llvm::any_of(currentFunctionTypes, [](Type type) {
    return isa<LLVM::LLVMArrayType, LLVM::LLVMStructType>(type);
  });
  bool needUpdatedResultType = isa<LLVM::LLVMStructType>(resultType);

  // If we need to update the calling convention and we have internal users,
  // emit a wrapper function as the structure of the function will have to
  // change.
  bool hasInternalUsers = !symbolUsers.getUsers(func).empty();
  if ((needUpdatedArgTypes || needUpdatedResultType) && hasInternalUsers)
    return emitCWrapper(func, aliasName, symtab);

  // Otherwise, we can update the function in place.
  updateExportedFunctionNameAndLinkage(func, aliasName, symbolUsers, symtab);

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
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addLegalOp<KGEN::CallSignatureOp>();
  target.addLegalOp<KGEN::CreateClosureOp>();
  target.addLegalOp<KGEN::LinkOp>();

  // Capture all the public symbols declared by kgen.export declarations.
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
  theModule->setAttrs(moduleAttrs.getDictionary(&getContext()));

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable &symtab = symtabAnalysis.getTopLevelSymbolTable();
  populateKGENToLLVMPatterns(typeConverter, patterns, symtab);
  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Process updates to any exported functions.
  mlir::SymbolUserMap symbolUsers(symtabAnalysis.getSymbolTables(), theModule);
  for (auto [sym, exportSymbol] : publicSymbols) {
    LLVM::LLVMFuncOp func = symtab.lookup<LLVM::LLVMFuncOp>(sym);

    // If we aren't exporting to C, we just need to update the name and linkage.
    if (!exportSymbol.isCExport) {
      updateExportedFunctionNameAndLinkage(func, exportSymbol.alias,
                                           symbolUsers, symtab);
      continue;
    }

    processCExportedFunction(func, exportSymbol.alias, symbolUsers, symtab);
  }

  // Convert the debug info within the IR.
  POPToLLVMDebugInfoTypeConverter debugTypeConverter(typeConverter);
  debugTypeConverter.applyRecursively(theModule);
}
