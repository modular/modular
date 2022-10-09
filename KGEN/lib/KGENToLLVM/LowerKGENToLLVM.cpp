//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
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

class ConvertKGENFunc : public mlir::ConvertOpToLLVMPattern<FuncOp> {
public:
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

    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        func.getLoc(), func.getNameAttr(), funcType,
        func.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private);
    // Set an attr to indicate that this thing is private. This is temporary -
    // we will end up removing the opaque wrappers.
    if (func.isPrivate())
      funcOp->setAttr("kgen_private", rewriter.getAttr<mlir::UnitAttr>());

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
// ConvertKGENPrecompiled
//===----------------------------------------------------------------------===//

/// Convert `kgen.precompiled.*` to an extern `llvm.func`.
template <typename PrecompiledOpT>
class ConvertKGENPrecompiled
    : public mlir::ConvertOpToLLVMPattern<PrecompiledOpT> {
public:
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
// ConvertKGENCall
//===----------------------------------------------------------------------===//

/// Convert `kgen.call` to `func.call` and re-use the latter's conversion to
/// LLVM.
class ConvertKGENCall : public mlir::ConvertOpToLLVMPattern<CallOp> {
public:
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

class ConvertKGENReturn : public mlir::ConvertOpToLLVMPattern<ReturnOp> {
public:
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

class ConvertKGENParamConstant
    : public mlir::ConvertOpToLLVMPattern<ParamConstantOp> {
public:
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

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       StructDeclarations &structDecls) {
  patterns.insert<
      // clang-format off
      ConvertKGENCall,
      ConvertKGENFunc,
      ConvertKGENPrecompiled<PrecompiledLLVMOp>,
      ConvertKGENPrecompiled<PrecompiledObjectOp>,
      ConvertKGENParamConstant,
      ConvertKGENReturn
      // clang-format on
      >(typeConverter);
  patterns.insert<
      // clang-format off
      ConvertKGENStructCreate,
      ConvertKGENStructExtract,
      ConvertKGENStructInsert
      // clang-format on
      >(typeConverter, structDecls);
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// Lower a concrete struct declaration to an LLVM struct. Struct types should
/// only appear in KGEN dialect operations.
static void configureTypeConverter(POPToLLVMTypeConverter &typeConverter,
                                   StructDeclarations &structDecls) {
  typeConverter.addConversion([&](RefType typeDef) -> Optional<Type> {
    auto it = structDecls.fieldTypes.find(typeDef.getName().getAttr());
    if (it == structDecls.fieldTypes.end()) {
      typeConverter.emitError("could not find struct declaration ")
          << typeDef.getName();
      return {};
    }
    // Substitute parameters into the field types.
    ParameterEvaluator evaluator;
    for (ParamBindAttr bind : typeDef.getParamValues())
      evaluator.setParameterValue(bind.getDecl(), bind.getValue());

    SmallVector<Type> elementTypes;
    for (Type type : it->second) {
      Type elementType =
          typeConverter.convertType(evaluator.getReboundType(type));
      if (!elementType) {
        typeConverter.emitError("failed to convert element type ") << type;
        return {};
      }
      elementTypes.push_back(elementType);
    }
    return LLVM::LLVMStructType::getLiteral(typeDef.getContext(), elementTypes);
  });
}

//===----------------------------------------------------------------------===//
// Emit C API Wrappers
//===----------------------------------------------------------------------===//

/// Recursively flatten a struct type into the function argument list. Pack the
/// struct from the flat arguments and return it.
static Value flattenArgumentStruct(ImplicitLocOpBuilder &b,
                                   LLVM::LLVMStructType structTy, Block *body) {
  Value result = b.create<LLVM::UndefOp>(structTy);
  for (auto &type : llvm::enumerate(structTy.getBody())) {
    Value value;
    if (auto nestedStruct = dyn_cast<LLVM::LLVMStructType>(type.value()))
      value = flattenArgumentStruct(b, nestedStruct, body);
    else
      value = body->addArgument(type.value(), b.getLoc());
    result = b.create<LLVM::InsertValueOp>(result, value, type.index());
  }
  return result;
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

/// Break up the structs in the given arguments and result type. Append new
/// arguments to `body` and populate `newArgs` with the packed structs created
/// at the top of the body. Return the slice of arguments that represent the
/// result arguments.
static ArrayRef<BlockArgument> breakUpStructs(Location loc, Block *body,
                                              ArrayRef<BlockArgument> args,
                                              Type resultTy,
                                              SmallVectorImpl<Value> &newArgs) {
  // Flatten structs in the argument list.
  ImplicitLocOpBuilder b(loc, loc.getContext());
  b.setInsertionPointToStart(body);
  for (Value arg : args) {
    b.setLoc(arg.getLoc());
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(arg.getType()))
      newArgs.push_back(flattenArgumentStruct(b, structTy, body));
    else
      newArgs.push_back(body->addArgument(arg.getType(), arg.getLoc()));
  }

  // Flatten the results if necessary at all the return points.
  ArrayRef<BlockArgument> results;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    unsigned numAdded = flattenResultStruct(loc, structTy, body);
    results = body->getArguments().take_back(numAdded);
  }

  return results;
}

/// Break up structs in the arguments and results of the given LLVM function
/// in-place. The function must be top-level as callsites are not modified.
static void breakUpStructsInPlace(LLVM::LLVMFuncOp func) {
  // If there are no structs to flatten, exit early.
  auto isStruct = [](Type type) { return type.isa<LLVM::LLVMStructType>(); };
  if (!llvm::any_of(func.getArgumentTypes(), isStruct) &&
      !isStruct(func.getResultTypes().front()))
    return;

  Block *entry = &func.getBody().front();
  Type resultTy = func.getResultTypes().front();
  SmallVector<Value> newArgs;
  ArrayRef<BlockArgument> results =
      breakUpStructs(func.getLoc(), entry, llvm::to_vector(func.getArguments()),
                     resultTy, newArgs);

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

/// Walk the terminal types of a struct with their positions in the struct.
static void
walkFlattenedStruct(LLVM::LLVMStructType structTy,
                    function_ref<void(Type, ArrayRef<LLVM::GEPArg>)> eachFn,
                    SmallVectorImpl<LLVM::GEPArg> &pos) {
  pos.emplace_back(nullptr);
  for (auto &type : llvm::enumerate(structTy.getBody())) {
    pos.back() = LLVM::GEPArg(type.index());
    if (auto nested = dyn_cast<LLVM::LLVMStructType>(type.value()))
      walkFlattenedStruct(nested, eachFn, pos);
    else
      eachFn(type.value(), pos);
  }
  pos.pop_back();
}
/// Walk the terminal types of the provided struct with their positions. This
/// also unwraps the top-level pointer, assuming that the value is of type
/// `!llvm.ptr<struct>`.
/// CAUTION: Do not use this to index into an array of structs! Use
/// `walkFlattenedStruct` directly for that.
static void walkFlattenedPtrToStruct(
    LLVM::LLVMStructType structTy,
    function_ref<void(Type, ArrayRef<LLVM::GEPArg>)> eachFn) {
  // This zero is to unwrap the top-level pointer.
  SmallVector<LLVM::GEPArg> pos = {0};
  walkFlattenedStruct(structTy, eachFn, pos);
}

/// Recursively pack the struct and put it in a pointer type.
// FIXME: LLVMStructType doesn't support replaceSubElements.
static LLVM::LLVMStructType recursivelyPack(LLVM::LLVMStructType structTy) {
  SmallVector<Type> body;
  body.reserve(structTy.getBody().size());
  for (Type type : structTy.getBody()) {
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type))
      body.push_back(recursivelyPack(structTy));
    else
      body.push_back(type);
  }
  return LLVM::LLVMStructType::getLiteral(structTy.getContext(), body, true);
}

/// Emit a wrapper function that takes opaque pointers to packed argument and
/// result structs.
static LLVM::LLVMFuncOp emitOpaqueWrapper(LLVM::LLVMFuncOp func,
                                          LLVM::LLVMFunctionType funcTy) {
  Block *entry = new Block;
  ImplicitLocOpBuilder b(func.getLoc(), func.getContext());
  b.setInsertionPointToStart(entry);

  // Unpack dense packed structs into a single flat argument. To make this
  // easier, we pack every argument into one struct.
  SmallVector<Type> flattenedTypes;
  flattenedTypes.reserve(func.getNumArguments());
  SmallVector<Value> callArgs;
  callArgs.reserve(func.getNumArguments());
  for (Type argTy : funcTy.getParams()) {
    auto structTy = dyn_cast<LLVM::LLVMStructType>(argTy);
    // Add a non-struct arg to the parameter pack.
    if (!structTy) {
      flattenedTypes.push_back(argTy);
      continue;
    }
    // Unpack the densely packed struct.
    flattenedTypes.push_back(recursivelyPack(structTy));
  }

  if (!flattenedTypes.empty()) {
    // Create the correct struct type. We always create a struct even if there's
    // only a single element in it because from a memory perspective, it doesn't
    // matter.
    LLVM::LLVMStructType allArgs =
        LLVM::LLVMStructType::getLiteral(b.getContext(), flattenedTypes, true);

    // One block argument for all the arguments. We'll walk this struct to get
    // the call arguments.
    Value arg =
        entry->addArgument(LLVM::LLVMPointerType::get(allArgs), func.getLoc());
    // Walk this struct and extract all the values to pass into the call.
    walkFlattenedPtrToStruct(
        allArgs, [&](Type type, ArrayRef<LLVM::GEPArg> pos) {
          Value curPtr =
              b.create<LLVM::GEPOp>(LLVM::LLVMPointerType::get(type), arg, pos);
          callArgs.push_back(b.create<LLVM::LoadOp>(curPtr));
        });
  }

  // The results are already flattened so just index into the provided pointer.
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(funcTy.getReturnType())) {
    Value ptr = entry->addArgument(
        LLVM::LLVMPointerType::get(recursivelyPack(structTy)), func.getLoc());
    walkFlattenedPtrToStruct(structTy,
                             [&](Type type, ArrayRef<LLVM::GEPArg> pos) {
                               callArgs.push_back(b.create<LLVM::GEPOp>(
                                   LLVM::LLVMPointerType::get(type), ptr, pos));
                             });
  }

  // Emit the call to the flattened version.
  assert(callArgs.size() == func.getNumArguments());
  auto call = b.create<LLVM::CallOp>(func, callArgs);

  // Check for a primitive result type.
  if (func.getNumResults())
    b.create<LLVM::ReturnOp>(call.getResult());
  else
    b.create<LLVM::ReturnOp>(ValueRange());

  // Create the function.
  auto newFuncTy =
      LLVM::LLVMFunctionType::get(func.getFunctionType().getReturnType(),
                                  llvm::to_vector(entry->getArgumentTypes()));
  b.clearInsertionPoint();
  auto newFunc = b.create<LLVM::LLVMFuncOp>(
      (func.getName() + "_opaque_wrapper").str(), newFuncTy, func.getLinkage());
  newFunc.getBody().push_back(entry);
  return newFunc;
}

/// Emit a wrapper for a function with structs broken up in the arguments and
/// results.
static LLVM::LLVMFuncOp emitCWrapper(LLVM::LLVMFuncOp func) {
  Block *entry = new Block;
  Type resultTy = func.getResultTypes().front();
  SmallVector<Value> newArgs;
  ArrayRef<BlockArgument> results = breakUpStructs(
      func.getLoc(), entry, func.getArguments(), resultTy, newArgs);

  // Create the nested call.
  ImplicitLocOpBuilder b(func.getLoc(), func.getContext());
  b.setInsertionPointToEnd(entry);
  auto call = b.create<LLVM::CallOp>(func, newArgs);

  // Unpack and store the results.
  SmallVector<Value> newResults;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    resultTy = LLVM::LLVMVoidType::get(func.getContext());
    unsigned idx = 0;
    flattenResultStruct(b, structTy, call.getResult(), results, idx);
  } else if (call.getNumResults()) {
    newResults.push_back(call.getResult());
  }
  b.create<LLVM::ReturnOp>(newResults);

  // Create the new function.
  auto funcTy = LLVM::LLVMFunctionType::get(
      resultTy, llvm::to_vector(entry->getArgumentTypes()));
  b.clearInsertionPoint();
  auto newFunc = b.create<LLVM::LLVMFuncOp>(
      (func.getName() + "_c_wrapper").str(), funcTy, func.getLinkage());
  newFunc.getBody().push_back(entry);
  return newFunc;
}

/// Break up argument and result structs in-place for the given top-level
/// funcs and emit C wrappers for specific non-top-level funcs.
static LogicalResult emitWrappers(ModuleOp theModule,
                                  ArrayRef<std::string> breakUpStructs,
                                  ArrayRef<std::string> emitCWrappers,
                                  bool emitOpaqueWrappers) {
  // Ensure that top-level funcs do not have callsites.
  llvm::StringMap<LLVM::CallOp> callsites;
  for (auto func : theModule.getOps<LLVM::LLVMFuncOp>())
    for (auto call : func.getOps<LLVM::CallOp>())
      if (Optional<StringRef> callee = call.getCallee())
        callsites.try_emplace(*callee, call);

  // Break up structs in-place in the specific top-level funcs.
  auto opaqueWrapperAttrName =
      StringAttr::get(theModule.getContext(), "opaque_wrapper");
  SymbolTable symtab(theModule);
  for (StringRef funcName : breakUpStructs) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(funcName);
    if (!func)
      return theModule.emitError("cannot find func: @") << funcName;
    // If the function's linkage is private, don't bother creating a wrapper.
    if (func->getAttr("kgen_private")) {
      mlir::emitWarning(
          func.getLoc(),
          "will not emit wrappers for this function marked private");
      continue;
    }

    if (auto it = callsites.find(funcName); it != callsites.end()) {
      return func.emitError("func is not top-level")
                 .attachNote(it->second.getLoc())
             << "callsite here";
    }
    if (func.isExternal())
      return func.emitError("cannot break up structs of an external function");

    LLVM::LLVMFunctionType funcTy = func.getFunctionType();
    breakUpStructsInPlace(func);
    if (emitOpaqueWrappers) {
      LLVM::LLVMFuncOp wrapper = emitOpaqueWrapper(func, funcTy);
      StringAttr wrapperRef = symtab.insert(wrapper, ++Block::iterator(func));
      func->setAttr(opaqueWrapperAttrName, FlatSymbolRefAttr::get(wrapperRef));
    }
  }

  // Emit C wrappers for the specific funcs.
  auto cWrapperAttrName = StringAttr::get(theModule.getContext(), "c_wrapper");
  for (StringRef funcName : emitCWrappers) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(funcName);
    if (!func)
      return theModule.emitError("cannot find func: @") << funcName;
    // If the function's linkage is private, don't bother creating a wrapper.
    if (func.getLinkage() == LLVM::Linkage::Private) {
      mlir::emitWarning(
          func.getLoc(),
          "will not emit wrappers for this function marked private");
      continue;
    }

    // Emit the wrapper, insert it and rename it if necessary, then store a
    // reference to the wrapper on the original function.
    LLVM::LLVMFuncOp wrapper = emitCWrapper(func);
    StringAttr wrapperRef = symtab.insert(wrapper, ++Block::iterator(func));
    func->setAttr(cWrapperAttrName, FlatSymbolRefAttr::get(wrapperRef));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

static void setStringListOption(Pass::ListOption<std::string> &opt,
                                ArrayRef<StringRef> values) {
  for (StringRef value : values)
    opt.push_back(value.str());
}

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENToLLVMPass
    : public KGEN::impl::LowerKGENToLLVMBase<LowerKGENToLLVMPass> {
  using LowerKGENToLLVMBase::LowerKGENToLLVMBase;

  explicit LowerKGENToLLVMPass(ArrayRef<StringRef> breakUpStructs,
                               ArrayRef<StringRef> emitCWrappers,
                               bool emitOpaqueWrappers) {
    setStringListOption(this->breakUpStructs, breakUpStructs);
    setStringListOption(this->emitCWrappers, emitCWrappers);
    this->emitOpaqueWrappers = emitOpaqueWrappers;
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
  configureTypeConverter(typeConverter, structDecls);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateKGENToLLVMPatterns(typeConverter, patterns, structDecls);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Break up structs in top-level funcs exposed to C.
  if (failed(emitWrappers(theModule, breakUpStructs, emitCWrappers,
                          emitOpaqueWrappers)))
    signalPassFailure();
}
