//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "LLVMLoweringUtils.h"
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

    // Create the LLVM function.
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        func.getLoc(), func.getNameAttr(), funcType,
        func.getVisibility() == mlir::SymbolTable::Visibility::Public
            ? LLVM::Linkage::External
            : LLVM::Linkage::Private);

    // And move the func's body into the new function.
    rewriter.inlineRegionBefore(func.getBodyRegion(0), funcOp.getBody(),
                                funcOp.end());
    (void)rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());

    // Remove the function.
    rewriter.eraseOp(func);
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
    if (auto dtype = op.getValue().dyn_cast<DTypeConstantAttr>()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, rewriter.getI8Type(), dtype.getDType().getValue());
    } else if (auto attr = op.getValue().dyn_cast<TypedAttr>()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, getTypeConverter()->convertType(attr.getType()), attr);
    } else {
      return rewriter.notifyMatchFailure(op, "unknown parameter value type");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaCastToBuiltin
//===----------------------------------------------------------------------===//

class ConvertMetaCastToBuiltin
    : public mlir::ConvertOpToLLVMPattern<MetaCastToBuiltinOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MetaCastToBuiltinOp op, MetaCastToBuiltinOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, opAdaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaCastFromBuiltin
//===----------------------------------------------------------------------===//

class ConvertMetaCastFromBuiltin
    : public mlir::ConvertOpToLLVMPattern<MetaCastFromBuiltinOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MetaCastFromBuiltinOp op,
                  MetaCastFromBuiltinOpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, opAdaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaPointerRebind
//===----------------------------------------------------------------------===//

/// A fully-specific pointer rebind between an unknown dtype and a known dtype
/// is converted to a bitcast.
class ConvertMetaPointerRebind
    : public mlir::ConvertOpToLLVMPattern<PointerRebindOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PointerRebindOp op, PointerRebindOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
          op, getTypeConverter()->convertType(op.getType()),
          adaptor.getInput());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferConstruct
//===----------------------------------------------------------------------===//

/// Convert the construction of a buffer to building the LLVM struct.
class ConvertMetaBufferConstruct
    : public mlir::ConvertOpToLLVMPattern<BufferConstructOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferConstructOp op, BufferConstructOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptorBuilder buffer(op.getType(), op.getLoc(), rewriter,
                                   *getTypeConverter());
    // Just return the pointer for a bare pointer buffer.
    if (buffer.isBarePtr()) {
      rewriter.replaceOp(op, adaptor.getPtr());
      return success();
    }

    Value buf = buffer.emitUndef();
    buf = buffer.emitSetPtr(buf, adaptor.getPtr());
    if (Value size = adaptor.getSize())
      buf = buffer.emitSetSize(buf, size);
    if (Value dtype = adaptor.getDType())
      buf = buffer.emitSetDType(buf, dtype);
    rewriter.replaceOp(op, buf);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferSize
//===----------------------------------------------------------------------===//

/// Convert the size of a buffer with a known size to a constant. Otherwise,
/// generate an `llvm.extractvalue`.
class ConvertMetaBufferSize
    : public mlir::ConvertOpToLLVMPattern<BufferSizeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferSizeOp op, BufferSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptorBuilder buffer(op.getValue(), op.getLoc(), rewriter,
                                   *getTypeConverter());
    rewriter.replaceOp(op, buffer.emitGetSize(adaptor.getValue()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferDType
//===----------------------------------------------------------------------===//

/// Convert the data type of a buffer with a known data type to a constant with
/// the value of the `DType::getValue` enum. Otherwise, generate an
/// `llvm.extractvalue`.
class ConvertMetaBufferDType
    : public mlir::ConvertOpToLLVMPattern<BufferDTypeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferDTypeOp op, BufferDTypeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptorBuilder buffer(op.getValue(), op.getLoc(), rewriter,
                                   *getTypeConverter());
    rewriter.replaceOp(op, buffer.emitGetDType(adaptor.getValue()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferAddress
//===----------------------------------------------------------------------===//

/// Convert the address of a buffer with known size and element type to itself,
/// since those buffers are converted to raw pointers. Otherwise, generate an
/// `llvm.extractvalue` of the pointer field.
class ConvertMetaBufferAddress
    : public mlir::ConvertOpToLLVMPattern<BufferAddressOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferAddressOp op, BufferAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptorBuilder buffer(op.getValue(), op.getLoc(), rewriter,
                                   *getTypeConverter());
    rewriter.replaceOp(op, buffer.emitGetPtr(adaptor.getValue()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMetaBufferRebind
//===----------------------------------------------------------------------===//

/// A buffer rebind can cast between a buffer with an unspecified size or
/// element type to one with specified size or element type. When that happens,
/// generate the necessary struct unpacking and repacking and bitcasts.
class ConvertMetaBufferRebind
    : public mlir::ConvertOpToLLVMPattern<BufferRebindOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferRebindOp op, BufferRebindOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BufferDescriptorBuilder in(op.getInput(), op.getLoc(), rewriter,
                               *getTypeConverter());
    BufferDescriptorBuilder out(op.getOutput(), op.getLoc(), rewriter,
                                *getTypeConverter());

    // If the input and output types are the same, fold away the op.
    if (in.getType() == out.getType()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Convert the pointer.
    Value outPtr = in.emitGetPtr(adaptor.getInput());
    if (in.getDType() != out.getDType()) {
      outPtr = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), getLLVMPointerTo(getContext(), out.getDType()), outPtr);
    }

    // Bare pointer output.
    if (out.isBarePtr()) {
      rewriter.replaceOp(op, outPtr);
      return success();
    }

    // Create the new struct.
    Value outBuffer = out.emitUndef();

    // If the output buffer has an unknown size, insert it.
    if (out.getSizeIndex()) {
      Value inSize = in.emitGetSize(adaptor.getInput());
      outBuffer = out.emitSetSize(outBuffer, inSize);
    }

    // If the output buffer has an unknown data type, insert it.
    if (out.getDTypeIndex()) {
      Value inDType = in.emitGetDType(adaptor.getInput());
      outBuffer = out.emitSetDType(outBuffer, inDType);
    }

    // Insert the casted pointer.
    rewriter.replaceOp(op, out.emitSetPtr(outBuffer, outPtr));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertKGENCall,
      ConvertKGENFunc,
      ConvertKGENParamConstant,
      ConvertKGENReturn,
      ConvertMetaBufferAddress,
      ConvertMetaBufferConstruct,
      ConvertMetaBufferDType,
      ConvertMetaBufferSize,
      ConvertMetaBufferRebind,
      ConvertMetaCastFromBuiltin,
      ConvertMetaCastToBuiltin,
      ConvertMetaPointerRebind
      // clang-format on
      >(typeConverter);
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
    if (auto nestedStruct = type.value().dyn_cast<LLVM::LLVMStructType>())
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
    if (auto nestedStruct = type.dyn_cast<LLVM::LLVMStructType>()) {
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
    if (auto nestedStruct = type.value().dyn_cast<LLVM::LLVMStructType>())
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
    if (auto structTy = arg.getType().dyn_cast<LLVM::LLVMStructType>())
      newArgs.push_back(flattenArgumentStruct(b, structTy, body));
    else
      newArgs.push_back(body->addArgument(arg.getType(), arg.getLoc()));
  }

  // Flatten the results if necessary at all the return points.
  ArrayRef<BlockArgument> results;
  if (auto structTy = resultTy.dyn_cast<LLVM::LLVMStructType>()) {
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
  if (auto structTy = resultTy.dyn_cast<LLVM::LLVMStructType>()) {
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
    if (auto nested = type.value().dyn_cast<LLVM::LLVMStructType>())
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
    if (auto structTy = type.dyn_cast<LLVM::LLVMStructType>())
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
    auto structTy = argTy.dyn_cast<LLVM::LLVMStructType>();
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
  if (auto structTy = funcTy.getReturnType().dyn_cast<LLVM::LLVMStructType>()) {
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
  if (auto structTy = resultTy.dyn_cast<LLVM::LLVMStructType>()) {
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
  SymbolTable symtab(theModule);
  auto opaqueWrapperAttrName =
      StringAttr::get(theModule.getContext(), "opaque_wrapper");
  for (StringRef funcName : breakUpStructs) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(funcName);
    if (!func)
      return theModule.emitError("cannot find func: @") << funcName;
    // If the function's linkage is private, don't bother creating a wrapper.
    if (func.getLinkage() == LLVM::Linkage::Private) {
      func.emitWarning(
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
      func.emitWarning(
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

namespace {
struct ConvertKGENToLLVMPass
    : public ConvertKGENToLLVMBase<ConvertKGENToLLVMPass> {
  explicit ConvertKGENToLLVMPass(ArrayRef<StringRef> breakUpStructs,
                                 ArrayRef<StringRef> emitCWrappers,
                                 bool emitOpaqueWrappers) {
    setStringListOption(this->breakUpStructs, breakUpStructs);
    setStringListOption(this->emitCWrappers, emitCWrappers);
    this->emitOpaqueWrappers = emitOpaqueWrappers;
  }

  void runOnOperation() override;
};
} // namespace

void ConvertKGENToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<KGENDialect, MetaDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(theModule->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateKGENToLLVMPatterns(typeConverter, patterns);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Break up structs in top-level funcs exposed to C.
  if (failed(emitWrappers(theModule, breakUpStructs, emitCWrappers,
                          emitOpaqueWrappers)))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass>
M::KGEN::createConvertKGENToLLVMPass(ArrayRef<StringRef> breakUpStructs,
                                     ArrayRef<StringRef> emitCWrappers,
                                     bool emitOpaqueWrappers) {
  return std::make_unique<ConvertKGENToLLVMPass>(breakUpStructs, emitCWrappers,
                                                 emitOpaqueWrappers);
}
