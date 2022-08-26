//===- ConvertKGENToLLVM.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
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
// ConvertKGENKernel
//===----------------------------------------------------------------------===//

class ConvertKGENKernel : public mlir::ConvertOpToLLVMPattern<KernelOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KernelOp kernel, KernelOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the kernel signature.
    TypeConverter::SignatureConversion result(kernel.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        kernel.getFunctionType(),
        /*isVariadic=*/false, result);
    if (!funcType)
      return emitError(kernel.getLoc(), "failed to convert kernel signature");

    // Create the LLVM function.
    auto funcOp = rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        kernel, kernel.getNameAttr(), funcType);

    // And move the kernel's body into the new function.
    rewriter.inlineRegionBefore(kernel.getBodyRegion(0), funcOp.getBody(),
                                funcOp.end());
    (void)rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());
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
      ConvertKGENKernel,
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

/// Recursively flatten a result struct type. Unpack the struct and store the
/// nested values into pointer arguments.
static void flattenResultStruct(ImplicitLocOpBuilder &b,
                                LLVM::LLVMStructType structTy, Value result,
                                Block *body) {
  for (auto &type : llvm::enumerate(structTy.getBody())) {
    Value value = b.create<LLVM::ExtractValueOp>(result, type.index());
    if (auto nestedStruct = type.value().dyn_cast<LLVM::LLVMStructType>()) {
      flattenResultStruct(b, nestedStruct, value, body);
    } else {
      Value ptr = body->addArgument(LLVM::LLVMPointerType::get(type.value()),
                                    b.getLoc());
      b.create<LLVM::StoreOp>(value, ptr);
    }
  }
}

/// Break up structs in the arguments and results of the given LLVM function.
/// For example, consider the following function that accepts and returns a
/// buffer:
///
/// ```mlir
/// llvm.func @slice(%a: !llvm.struct<(i64, ptr<f32>)>, %i: !llvm.ptr<i64>)
///     -> !llvm.struct<(i64, ptr<f32>)>
/// ```
///
/// The resulting signature will be:
///
/// ```mlir
/// llvm.func @slice(%a_0: i64, %a_1: !llvm.ptr<f32>, %i: !llvm.ptr<i64>,
///                  %res_0: !llvm.ptr<i64>, %res_1: !llvm.ptr<ptr<f32>>)
///     -> !llvm.void
/// ```
///
static void breakUpStructs(LLVM::LLVMFuncOp func) {
  // If there are no structs to flatten, exit early.
  auto isStruct = [](Type type) { return type.isa<LLVM::LLVMStructType>(); };
  if (!llvm::any_of(func.getArgumentTypes(), isStruct) &&
      !isStruct(func.getResultTypes().front()))
    return;

  // Create the wrapper body.
  ImplicitLocOpBuilder b(func.getLoc(), func.getContext());
  Block *body = &func.getBody().front();
  SmallVector<Value> args;

  // Flatten structs in the argument list.
  b.setInsertionPointToStart(body);
  for (Value arg : llvm::to_vector(func.getArguments())) {
    b.setLoc(arg.getLoc());
    if (auto structTy = arg.getType().dyn_cast<LLVM::LLVMStructType>())
      args.push_back(flattenArgumentStruct(b, structTy, body));
    else
      args.push_back(body->addArgument(arg.getType(), b.getLoc()));
    arg.replaceAllUsesWith(args.back());
  }

  // Erase the old arguments.
  llvm::BitVector indices(body->getNumArguments());
  indices.set(0, func.getNumArguments());
  body->eraseArguments(indices);

  // Flatten the results if necessary at all the return points.
  Type resultTy = func.getNumResults() ? func.getResultTypes().front()
                                       : b.getType<LLVM::LLVMVoidType>();
  if (auto structTy = resultTy.dyn_cast<LLVM::LLVMStructType>()) {
    resultTy = b.getType<LLVM::LLVMVoidType>();
    for (auto ret : llvm::make_early_inc_range(func.getOps<LLVM::ReturnOp>())) {
      flattenResultStruct(b, structTy, ret.getOperand(0), body);
      b.setInsertionPoint(ret);
      b.create<LLVM::ReturnOp>(ValueRange());
      ret->erase();
    }
  }

  // Update the function type.
  func.setFunctionTypeAttr(TypeAttr::get(LLVM::LLVMFunctionType::get(
      resultTy, llvm::to_vector(body->getArgumentTypes()))));
}

/// Emit C wrappers for all LLVM functions in the given module.
static LogicalResult breakUpStructs(ModuleOp theModule,
                                    ArrayRef<std::string> topLevelKernels) {
  // Ensure that top-level kernels do not have callsites.
  llvm::StringMap<LLVM::CallOp> callsites;
  for (auto func : theModule.getOps<LLVM::LLVMFuncOp>())
    for (auto call : func.getOps<LLVM::CallOp>())
      if (Optional<StringRef> callee = call.getCallee())
        callsites.try_emplace(*callee, call);

  SymbolTable symtab(theModule);
  for (StringRef topLevelKernel : topLevelKernels) {
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(topLevelKernel);
    if (auto it = callsites.find(topLevelKernel); it != callsites.end()) {
      return func.emitError("kernel is not top-level")
                 .attachNote(it->second.getLoc())
             << "callsite here";
    }
    if (func.isExternal())
      return func.emitError("cannot break up structs of an external function");

    breakUpStructs(func);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKGENToLLVMPass
    : public ConvertKGENToLLVMBase<ConvertKGENToLLVMPass> {
  explicit ConvertKGENToLLVMPass(ArrayRef<StringRef> topLevelKernels) {
    SmallVector<std::string> names;
    names.reserve(topLevelKernels.size());
    for (StringRef topLevelKernel : topLevelKernels)
      names.push_back(topLevelKernel.str());
    this->topLevelKernels = names;
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

  // Break up structs in top-level kernels exposed to C.
  if (failed(breakUpStructs(theModule, topLevelKernels)))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass>
M::KGEN::createConvertKGENToLLVMPass(ArrayRef<StringRef> topLevelKernels) {
  return std::make_unique<ConvertKGENToLLVMPass>(topLevelKernels);
}
