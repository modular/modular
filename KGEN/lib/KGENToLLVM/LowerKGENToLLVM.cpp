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

    LLVM::Linkage llvmLinkage;
    switch (func.getLinkage()) {
    case Linkage::Public:
      llvmLinkage = LLVM::Linkage::External;
      break;
    case Linkage::ModulePrivate:
      llvmLinkage = LLVM::Linkage::Internal;
      break;
    case Linkage::LibraryPrivate:
      llvmLinkage = LLVM::Linkage::Linkonce;
      break;
    }

    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        func.getLoc(), func.getNameAttr(), funcType, llvmLinkage);

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
// ConvertKGENExtern
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
    rewriter.template replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, op.getNameAttr(), funcType, LLVM::Linkage::External);

    return success();
  }
};

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
    rewriter.template replaceOpWithNewOp<LLVM::GlobalOp>(
        op, llvmType, false, LLVM::Linkage::External, op.getName(),
        /*value=*/nullptr);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENPrecompiled
//===----------------------------------------------------------------------===//

/// Convert `kgen.precompiled.*` to an extern `llvm.func`.
template <typename PrecompiledOpT>
struct ConvertKGENPrecompiled
    : public mlir::ConvertOpToLLVMPattern<PrecompiledOpT> {
  using mlir::ConvertOpToLLVMPattern<PrecompiledOpT>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PrecompiledOpT op, typename PrecompiledOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(op.getNumArguments());
    Type funcType = this->getTypeConverter()->convertFunctionSignature(
        op.getFunctionType(),
        /*isVariadic=*/false, result);
    if (!funcType)
      return emitError(op.getLoc(), "failed to convert func signature");

    // Replace it with an LLVM function that has no body.
    rewriter.template replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, op.getNameAttr(), funcType, LLVM::Linkage::External);

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

/// Convert `kgen.call` to `func.call` and re-use the latter's conversion to
/// LLVM.
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
    } else if (auto attr = dyn_cast<TypedAttr>(op.getValue())) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, getTypeConverter()->convertType(attr.getType()), attr);
    } else {
      return rewriter.notifyMatchFailure(op, "unknown parameter value type");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructOp
//===----------------------------------------------------------------------===//

/// Information about a struct declaration.
struct StructDeclarations {
  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int64_t> fieldIndices;

  /// A map from struct name to field types. Used for type conversions.
  DenseMap<StringAttr, SmallVector<Type>> fieldTypes;
};

/// Struct operations need to refer to the struct declaration symbol.
class ConvertKGENStructOpBase {
public:
  explicit ConvertKGENStructOpBase(StructDeclarations &structDecls)
      : structDecls(structDecls) {}

  /// Get the index of the struct field.
  Optional<int64_t> getFieldIndex(StringAttr name, RefType typeDef) const {
    auto it =
        structDecls.fieldIndices.find({typeDef.getName().getAttr(), name});
    if (it == structDecls.fieldIndices.end())
      return {};
    return it->second;
  }

private:
  StructDeclarations &structDecls;
};

template <typename StructOp>
struct ConvertKGENStructOp : public mlir::ConvertOpToLLVMPattern<StructOp>,
                             public ConvertKGENStructOpBase {
  ConvertKGENStructOp(mlir::LLVMTypeConverter &typeConverter,
                      StructDeclarations &structDecls)
      : mlir::ConvertOpToLLVMPattern<StructOp>(typeConverter),
        ConvertKGENStructOpBase(structDecls) {}
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructCreate
//===----------------------------------------------------------------------===//

struct ConvertKGENStructCreate : public ConvertKGENStructOp<StructCreateOp> {
  using ConvertKGENStructOp::ConvertKGENStructOp;

  LogicalResult
  matchAndRewrite(StructCreateOp op, StructCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = dyn_cast_if_present<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(op.getType()));
    if (!type)
      return failure();

    Value container = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
    for (auto &operand : llvm::enumerate(adaptor.getOperands()))
      container = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), container, operand.value(), operand.index());
    rewriter.replaceOp(op, container);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructInsert
//===----------------------------------------------------------------------===//

struct ConvertKGENStructInsert : public ConvertKGENStructOp<StructInsertOp> {
  using ConvertKGENStructOp::ConvertKGENStructOp;

  LogicalResult
  matchAndRewrite(StructInsertOp op, StructInsertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Optional<int64_t> index =
        getFieldIndex(op.getFieldAttr(), op.getContainer().getType());
    if (!index)
      return op.emitError("could not find struct declaration");
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getContainer(), adaptor.getValue(), *index);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructExtract
//===----------------------------------------------------------------------===//

struct ConvertKGENStructExtract : public ConvertKGENStructOp<StructExtractOp> {
  using ConvertKGENStructOp::ConvertKGENStructOp;

  LogicalResult
  matchAndRewrite(StructExtractOp op, StructExtractOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Optional<int64_t> index =
        getFieldIndex(op.getFieldAttr(), op.getContainer().getType());
    if (!index)
      return op.emitError("could not find struct declaration");
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getContainer(), *index);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructGEP
//===----------------------------------------------------------------------===//

struct ConvertKGENStructGEP : public ConvertKGENStructOp<StructGEPOp> {
  using ConvertKGENStructOp::ConvertKGENStructOp;

  LogicalResult
  matchAndRewrite(StructGEPOp op, StructGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type structType = op.getContainer().getType().getResolvedElementType();
    Optional<int64_t> index =
        getFieldIndex(op.getFieldAttr(), cast<RefType>(structType));
    if (!index)
      return op.emitError("could not find struct declaration");
    Type ptrType = getTypeConverter()->convertType(op.getType());
    if (!ptrType)
      return op.emitError("failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrType, adaptor.getContainer(), ArrayRef<LLVM::GEPArg>{0, *index});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       StructDeclarations &structDecls) {
  patterns.insert<
      // clang-format off
      ConvertKGENAddressOf,
      ConvertKGENCall,
      ConvertKGENFunc,
      ConvertKGENExternFunc,
      ConvertKGENExternVariable,
      ConvertKGENPrecompiled<PrecompiledLLVMOp>,
      ConvertKGENParamConstant,
      ConvertKGENReturn
      // clang-format on
      >(typeConverter);
  patterns.insert<
      // clang-format off
      ConvertKGENStructCreate,
      ConvertKGENStructExtract,
      ConvertKGENStructGEP,
      ConvertKGENStructInsert
      // clang-format on
      >(typeConverter, structDecls);
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// Replace a KGEN struct with an LLVM struct.
static LLVM::LLVMStructType
substituteStructDecl(const StructDeclarations &structDecls, RefType typeDef,
                     function_ref<Type(Type)> transformElement) {
  auto it = structDecls.fieldTypes.find(typeDef.getName().getAttr());
  if (it == structDecls.fieldTypes.end())
    return {};
  // Substitute parameters into the field types.
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : typeDef.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  SmallVector<Type> elementTypes;
  for (Type type : it->second) {
    Type elementType = transformElement(evaluator.getReboundType(type));
    if (!elementType)
      return {};
    elementTypes.push_back(elementType);
  }
  return LLVM::LLVMStructType::getLiteral(typeDef.getContext(), elementTypes);
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
    if (func.getLinkage() == LLVM::Linkage::Internal) {
      mlir::emitWarning(
          func.getLoc(),
          "will not rewrite calling convention for private functions");
      continue;
    }
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

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);

  // Collect all struct declarations and erase them.
  StructDeclarations structDecls;
  for (auto decl :
       llvm::make_early_inc_range(theModule.getOps<StructDeclOp>())) {
    SmallVector<Type> fieldTypes;
    for (auto &field : llvm::enumerate(decl.getFieldDecls())) {
      fieldTypes.push_back(field.value().getType());
      structDecls.fieldIndices.try_emplace(
          {decl.getNameAttr(), field.value().getNameAttr()}, field.index());
    }
    structDecls.fieldTypes.try_emplace(decl.getNameAttr(),
                                       std::move(fieldTypes));
    decl->erase();
  }

  // Configure the type converter.
  POPToLLVMTypeConverter typeConverter(theModule->getLoc(), options);
  typeConverter.addConversion([&](RefType typeDef) -> Optional<Type> {
    return substituteStructDecl(structDecls, typeDef, [&](Type elType) {
      return typeConverter.convertType(elType);
    });
  });

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateKGENToLLVMPatterns(typeConverter, patterns, structDecls);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Break up structs in top-level funcs exposed to C.
  if (failed(emitWrappers(theModule, topLevelFuncs)))
    return signalPassFailure();

  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types.
  std::function<Type(Type)> substituteRefs = [&](Type type) -> Type {
    if (auto ref = dyn_cast<RefType>(type))
      return substituteStructDecl(structDecls, ref, substituteRefs);
    auto itf = dyn_cast<mlir::SubElementTypeInterface>(type);
    if (!itf)
      return type;
    return itf.replaceSubElements([&](Type type) -> Type {
      if (auto ref = dyn_cast<RefType>(type))
        return substituteStructDecl(structDecls, ref, substituteRefs);
      return type;
    });
  };
  WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
    // Substitute any references in attributes.
    op->setAttrs(op->getAttrDictionary()
                     .replaceSubElements(substituteRefs)
                     .cast<DictionaryAttr>());

    // Substitute the result types.
    for (OpResult result : op->getOpResults()) {
      Type replType = substituteRefs(result.getType());
      if (!replType)
        return op->emitError("failed to substitute result type #")
               << result.getResultNumber();
      result.setType(replType);
    }

    // Substitute the block argument types.
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          Type replType = substituteRefs(arg.getType());
          if (!replType)
            return op->emitError("failed to substitute block argument type ")
                   << arg.getType();
          arg.setType(replType);
        }
      }
    }

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
