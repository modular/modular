//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLVMLoweringUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/Threading.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

/// Get the LLVM linkage kind for an export kind.
static LLVM::Linkage getLinkageKind(ExportKind exportKind) {
  switch (exportKind) {
  case ExportKind::NotExported:
    return LLVM::Linkage::Internal;
  case ExportKind::Exported:
  case ExportKind::CExported:
  case ExportKind::PackageExported:
    return LLVM::Linkage::External;
  }
  llvm_unreachable("invalid export kind");
}

namespace {

template <typename T>
static Attribute arrayAttrToDenseArrayAttr(Builder builder,
                                           POP::ArrayAttr array) {
  SmallVector<T> values =
      llvm::map_to_vector(array.getValues(), [](Attribute attr) -> T {
        if (auto integerAttr = ::dyn_cast<IntegerAttr>(attr))
          return static_cast<T>(integerAttr.getInt());
        return static_cast<T>(::cast<POP::SIMDAttr>(attr)
                                  .getValues()
                                  .front()
                                  .getIntVal()
                                  .getSExtValue());
      });
  if constexpr (std::is_same_v<T, int8_t>)
    return builder.getDenseI8ArrayAttr(values);
  else if constexpr (std::is_same_v<T, int16_t>)
    return builder.getDenseI16ArrayAttr(values);
  else if constexpr (std::is_same_v<T, int32_t>)
    return builder.getDenseI32ArrayAttr(values);
  else
    return builder.getDenseI64ArrayAttr(values);
}

static ErrorOrSuccess addArrayAttrToDict(Builder builder, NamedAttrList &attrs,
                                         StringRef name, POP::ArrayAttr array,
                                         mlir::Type elementType) {
  if (auto type = dyn_cast<POP::SIMDType>(elementType)) {
    if (type.getResolvedSize().value_or(-1) != 1)
      return Error("ArrayAttr elements must be a scalar");

    std::optional<KGENDType> dtype = type.getResolvedDType();

    if (!dtype)
      return Error("unable to resolve the dtype for the SIMD value");

    if (!dtype->isInt())
      return Error("ArrayAttr must be an integral dtype");

    return addArrayAttrToDict(
        builder, attrs, name, array,
        getEquivalentIntegerType(builder.getContext(), *dtype));
  }

  if (auto type = dyn_cast<IntegerType>(elementType);
      type && llvm::is_contained({8, 16, 32, 64}, type.getWidth())) {
    if (type.getWidth() == 8)
      attrs.append(name, arrayAttrToDenseArrayAttr<int8_t>(builder, array));
    else if (type.getWidth() == 16)
      attrs.append(name, arrayAttrToDenseArrayAttr<int16_t>(builder, array));
    else if (type.getWidth() == 32)
      attrs.append(name, arrayAttrToDenseArrayAttr<int32_t>(builder, array));
    else
      attrs.append(name, arrayAttrToDenseArrayAttr<int64_t>(builder, array));
    return success();
  }

  return Error("non-integral dtypes are not supported");
}

//===----------------------------------------------------------------------===//
// ConvertKGENFunc
//===----------------------------------------------------------------------===//

namespace {
/// Cached attribute identifiers.
struct AttributeIdentifiers {
  StringAttr noalias, noundef, nonnull;
};
} // namespace

/// Convert LLVM metadata expressed in KGEN attributes to an LLVM dialect
/// compatible representation. Unsupport metadata values are rejected.
static LogicalResult convertLLVMMetadata(LLVM::LLVMFuncOp func,
                                         SignatureType sig,
                                         DictionaryAttr metadata,
                                         const AttributeIdentifiers &ids) {
  NamedAttrList attrs = func->getAttrDictionary();
  SmallVector<Attribute> passthrough =
      llvm::to_vector(func.getPassthroughAttr());
  Builder b(func.getContext());

  for (const NamedAttribute &attr : metadata) {
    // Treat `llvm.*` metadata attributes as passthrough function attributes.
    Attribute value = attr.getValue();
    if (isa<LLVM::LLVMDialect>(attr.getNameDialect())) {
      StringAttr name = b.getStringAttr(
          attr.getName().strref().drop_front(StringRef("llvm.").size()));
      if (isa<mlir::UnitAttr>(value)) {
        // Add the metadata attribute name without the prefix.
        passthrough.push_back(name);
      } else if (auto intVal = dyn_cast<IntegerAttr>(value)) {
        // The LLVM exporter apparently expects the integer to be encoded as a
        // string. Push the pair as an array attribute.
        SmallVector<char> str;
        intVal.getValue().toString(str, /*Radix=*/10, /*Signed=*/true);
        passthrough.push_back(b.getArrayAttr(
            {name, b.getStringAttr(StringRef(str.data(), str.size()))}));
      } else if (auto str = dyn_cast<StringAttr>(value)) {
        // Strip the type from string attributes.
        passthrough.push_back(
            b.getArrayAttr({name, b.getStringAttr(str.getValue())}));
      } else {
        return mlir::emitError(func.getLoc(),
                               "unsupported LLVM passthrough attribute kind: ")
               << value;
      }
      continue;
    }

    // For anything else, forward them as function attributes.
    if (isa<mlir::UnitAttr, IntegerAttr>(value)) {
      // Propagate unit and integer attribute.
      attrs.append(attr.getName(), value);
    } else if (auto str = dyn_cast<StringAttr>(value)) {
      // Strip the type from string attributes.
      attrs.append(attr.getName(), b.getStringAttr(str.getValue()));
    } else if (auto array = dyn_cast<POP::ArrayAttr>(value)) {
      mlir::Type elementType = array.getType().getElementType();
      if (auto err =
              addArrayAttrToDict(b, attrs, attr.getName(), array, elementType);
          err.isError())
        return mlir::emitError(func.getLoc(), "unsupported array type: ")
               << array << " because " << err.takeError();
    } else if (auto array = dyn_cast<mlir::DenseI32ArrayAttr>(value)) {
      attrs.append(attr.getName(), array);
    } else {
      return mlir::emitError(func.getLoc(),
                             "unsupported LLVM metadata attribute kind: ")
             << value;
    }
  }

  // For each argument and result, leverage signature information to generate
  // the correpsonding LLVM argument and result attributes.
  SmallVector<Attribute> argAttrs;
  NamedAttrList list;
  for (auto [i, conv, type] :
       llvm::enumerate(sig.getArgConventions(), sig.getArguments())) {
    list.clear();
    // `exclusive` pointer implies `noalias` pointer argument.
    if (auto ptr = dyn_cast<PointerType>(type);
        ptr && cast<BoolAttr>(ptr.getExclusive()).getValue())
      list.set(ids.noalias, b.getUnitAttr());

    switch (conv) {
    case ArgConvention::OwnedInMem:
    case ArgConvention::InOut:
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
    case ArgConvention::InitSelf:
      // The compiler enforces that each function can only have one mutable
      // reference to an object at a time. Thus, we know the pointers that back
      // mutable in-memory arguments are noalias.
      list.set(ids.noalias, b.getUnitAttr());
      [[fallthrough]];

    case ArgConvention::Ref:
      // TODO(MOCO-914): `ref` arguments could be mutable references, but we
      // don't have the information in the IR anymore.
    case ArgConvention::BorrowedInMem:
      // We know the pointers that back in-memory arguments are nonnull.
      list.set(ids.nonnull, b.getUnitAttr());
      [[fallthrough]];

    case ArgConvention::BorrowedInReg:
    case ArgConvention::OwnedInReg:
      // The only thing we can say about values passed in-register is `noundef`,
      // which is equivalent to saying that they are known initialized. This
      // also applies to all the pointers passed for in-memory arguments.
      list.set(ids.noundef, b.getUnitAttr());
      break;
    }

    argAttrs.push_back(list.getDictionary(b.getContext()));
  }

  // Update the attributes.
  attrs.set(func.getArgAttrsAttrName(), b.getArrayAttr(argAttrs));
  func->setAttrs(attrs.getDictionary(func.getContext()));
  func.setPassthroughAttr(b.getArrayAttr(passthrough));
  return success();
}

/// Convert inline level to an LLVM passthrough attribute.
/// compatible representation. Unsupport metadata values are rejected.
static void convertInlineLevel(LLVM::LLVMFuncOp func, InlineLevel inlineLevel) {
  if (inlineLevel == InlineLevel::Automatic)
    return;

  SmallVector<Attribute> passthrough =
      llvm::to_vector(func.getPassthroughAttr());
  Builder b(func.getContext());

  const char *attrName;
  switch (inlineLevel) {
  case InlineLevel::Always:
  case InlineLevel::AlwaysNoDebug:
    attrName = "alwaysinline";
    break;
  case InlineLevel::Never:
    attrName = "noinline";
    break;
  default:
    llvm_unreachable("invalid InlineLevel enum");
  }
  passthrough.push_back(b.getStringAttr(attrName));
  func.setPassthroughAttr(b.getArrayAttr(passthrough));
}

/// Returns true if type is an empty !llvm.struct type, or an array of empty
/// types e.g !llvm.array<0 x ..any type..>, !llvm.array<N x empty_struct>
/// TODO: Consider querying size from DataLayout instead.
static bool isEmptyType(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](LLVM::LLVMArrayType arrayType) {
        if (arrayType.getNumElements() == 0)
          return true;
        return isEmptyType(arrayType.getElementType());
      })
      .Case([](LLVM::LLVMStructType structType) {
        bool emptyType = true;
        for (Type innerType : structType.getBody())
          emptyType &= isEmptyType(innerType);
        return emptyType;
      })
      .Default([](Type /* default */) { return false; });
}

/// Drops empty struct arguments from funcOp and replace usage with an undef
/// struct.
static void dropEmptyStructArguments(LLVM::LLVMFuncOp &func,
                                     ConversionPatternRewriter &rewriter) {
  SmallVector<unsigned> emptyArgIdx, nonEmptyArgIdx;
  SmallVector<Type> emptyArgType, nonEmptyArgTypes;
  for (auto [idx, argType] : enumerate(func.getArgumentTypes())) {
    if (isEmptyType(argType)) {
      emptyArgIdx.push_back(idx);
      emptyArgType.push_back(argType);
    } else {
      nonEmptyArgIdx.push_back(idx);
      nonEmptyArgTypes.push_back(argType);
    }
  }

  if (emptyArgIdx.empty())
    return;

  // If it has a body block erase empty struct function arguments and
  // replace their inner usage with undef empty struct types.
  if (!func.getBody().empty()) {
    Block *entryBlock = &func.getBody().front();
    rewriter.setInsertionPointToStart(entryBlock);
    TypeConverter::SignatureConversion sigConverter(func.getNumArguments());
    for (auto [idx, type] : zip(nonEmptyArgIdx, nonEmptyArgTypes))
      sigConverter.addInputs(idx, type);

    for (auto [idx, type] : zip(emptyArgIdx, emptyArgType)) {
      Value emtpyStruct = rewriter.create<LLVM::UndefOp>(func->getLoc(), type);
      sigConverter.remapInput(idx, emtpyStruct);
    }
    rewriter.applySignatureConversion(&func.getBody().front(), sigConverter);
  }

  // Update funcOp type.
  rewriter.modifyOpInPlace(func, [&]() {
    func.setType(LLVM::LLVMFunctionType::get(
        func.getFunctionType().getReturnType(), nonEmptyArgTypes));
  });
}

class ConvertKGENFunc : public ConvertSymbolOpToLLVM<FuncOp> {
public:
  ConvertKGENFunc(mlir::LLVMTypeConverter &tc, SymbolTable &symtab,
                  const AttributeIdentifiers &ids)
      : ConvertSymbolOpToLLVM(tc, symtab), ids(ids) {}

  LogicalResult matchAndRewrite(FuncOp func, FuncOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Convert the func signature.
    TypeConverter::SignatureConversion result(func.getNumArguments());
    Type funcType = getTypeConverter()->convertFunctionSignature(
        func.getFunctionType(), /*isVariadic=*/false,
        getTypeConverter()->getOptions().useBarePtrCallConv, result);
    if (!funcType)
      return emitError(func.getLoc(), "failed to convert func signature");

    TargetInfoAttr target = getTypeConverter()->getTarget();

    // Mark all functions as internal for now - we'll clean this up later.
    auto funcOp =
        createLLVMFunc(b, target, func.getLoc(), func.getNameAttr(), funcType,
                       getLinkageKind(func.getExportKind()));
    if (failed(convertLLVMMetadata(funcOp, func.getSignature(),
                                   func.getLLVMMetadataAttr(), ids)))
      return failure();
    if (func.isExported()) {
      funcOp.setDsoLocal(true);

      // Exported functions to the NVVM target get a special metadata
      // attribute to tell LLVM that these are kernel functions.
      if (llvm::is_contained({llvm::Triple::nvptx, llvm::Triple::nvptx64},
                             target.getTriple().getArch()))
        funcOp->setAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName(),
                        b.getUnitAttr());
    }

    if (func.getCoroutineType()) {
      Type coroType =
          typeConverter->convertType(func.getCoroutineType().value());
      funcOp->setAttr(func.getCoroutineTypeAttrName(), TypeAttr::get(coroType));
    }

    // Propagate InlineLevel as a passthrough LLVM attribute.
    convertInlineLevel(funcOp, func.getInlineLevel());

    // And move the func's body into the new function.
    b.inlineRegionBefore(func.getBodyRegion(), funcOp.getBody(), funcOp.end());
    (void)b.convertRegionTypes(&funcOp.getBody(), *getTypeConverter());

    // Drop empty struct arguments.
    dropEmptyStructArguments(funcOp, b);

    // Remove the function.
    symtab.remove(func);
    Block::iterator insertPt(func->getNextNode());
    funcOp->remove();
    symtab.insert(funcOp, insertPt);
    b.eraseOp(func);
    return success();
  }

private:
  const AttributeIdentifiers &ids;
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

    // Drop empty struct argument.
    auto filteredOperands = to_vector(
        llvm::make_filter_range(adaptor.getOperands(), [](Value operand) {
          return !isEmptyType(operand.getType());
        }));

    // Create the LLVM call operation.
    LLVM::CallOp llvmCall = createLLVMCall(rewriter, op.getLoc(), types,
                                           flatSymbol, filteredOperands);

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

// Helper for ConvertKGENParamConstant and ConvertKGENParamMaterialize.  Emit
// a good error when the type to be materialized is a non-materializable
// literal.
static void failLiteralMaterialization(Type t, ImplicitLocOpBuilder b) {
  if (isa<KGEN::IntLiteralType>(t))
    b.emitError("can't materialize IntLiteral in dynamic context");
  else if (isa<KGEN::FloatLiteralType>(t))
    b.emitError("can't materialize FloatLiteral in dynamic context");
  return;
}

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
    if (!value) {
      failLiteralMaterialization(op.getType(), b);
      return failure();
    }
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
    if (!value) {
      failLiteralMaterialization(op.getType(), b);
      return failure();
    }
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
// ConvertKGENRebind
//===----------------------------------------------------------------------===//

/// Intercept unfolded rebind ops to give a better error message.
struct ConvertKGENRebind : public ConvertPOPToLLVMPattern<RebindOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(RebindOp op, RebindOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get llvm types to compare because something like
    // !kgen.pointer<index> is same as !kgen.pointer<scalar<index>> when
    // lowered to llvm and should be allowed.
    Type resultType = getTypeConverter()->convertType(op.getType());
    Type inputType = getTypeConverter()->convertType(op.getInput().getType());
    rewriter.replaceOp(op, op.getInput());

    if (resultType != inputType) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << op.getInput().getType() << " to " << op.getType();
      return emitError(op.getLoc(),
                       "invalid rebind between two unequal types: " + str);
    }
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
// ConvertKGENStructCreate
//===----------------------------------------------------------------------===//

struct ConvertKGENStructCreate : ConvertPOPToLLVMPattern<StructCreateOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, StructCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type structType = convertType(op.getType());
    if (!structType)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "failed to convert struct type");
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value container =
        materializeLLVMStruct(b, structType, adaptor.getOperands());
    rewriter.replaceOp(op, container);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructReplace
//===----------------------------------------------------------------------===//

struct ConvertKGENStructReplace : ConvertPOPToLLVMPattern<StructReplaceOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructReplaceOp op, StructReplaceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getContainer(), adaptor.getValue(),
        op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructGet
//===----------------------------------------------------------------------===//

struct ConvertKGENStructGet : ConvertPOPToLLVMPattern<StructExtractOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, StructExtractOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getContainer(), op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENStructGEP
//===----------------------------------------------------------------------===//

struct ConvertKGENStructGEP : ConvertPOPToLLVMPattern<StructGEPOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructGEPOp op, StructGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PointerType ptrType = cast<PointerType>(op.getContainer().getType());
    Type elementType = convertType(ptrType.getElementType());
    if (!elementType)
      return op.emitError("failed to convert result type");
    LLVM::LLVMPointerType opaquePtr = LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, opaquePtr, elementType, adaptor.getContainer(),
        ArrayRef<LLVM::GEPArg>{
            0, static_cast<int32_t>(op.getIndexAttr().getInt())});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENVariantCreate
//===----------------------------------------------------------------------===//

struct ConvertKGENVariantCreate
    : public ConvertPOPToLLVMPattern<VariantCreateOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantCreateOp op, VariantCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto variantType =
        dyn_cast_if_present<LLVM::LLVMStructType>(convertType(op.getType()));
    if (!variantType)
      return failure();

    VariantHelper helper(rewriter, op.getLoc(), *getTypeConverter());
    Value result = helper.materializeLLVMVariant(
        variantType, adaptor.getOperand(), op.getIndex());
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENVariantIs
//===----------------------------------------------------------------------===//

/// Lower `kgen.variant.is` to an extract and integer compare.
struct ConvertKGENVariantIs : public ConvertPOPToLLVMPattern<VariantIsOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantIsOp op, VariantIsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value discr = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 1);
    auto variantType =
        cast<LLVM::LLVMStructType>(adaptor.getVariant().getType());
    Value discrVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), variantType.getBody().back(), op.getIndex());
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              discr, discrVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENVariantGet
//===----------------------------------------------------------------------===//

struct ConvertKGENVariantGet : ConvertPOPToLLVMPattern<VariantTakeOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantTakeOp op, VariantTakeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type valueType = convertType(op.getType());
    if (!valueType)
      return failure();
    auto variantType =
        cast<LLVM::LLVMStructType>(adaptor.getVariant().getType());
    auto contentType = cast<LLVM::LLVMArrayType>(variantType.getBody().front());

    // Extract the content and put it in the block of memory.
    Value content = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 0);

    SmallVector<Value> storageValues;
    for (unsigned i = 0, e = contentType.getNumElements(); i != e; ++i)
      storageValues.push_back(
          rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), content, i));

    VariantHelper helper(rewriter, op.getLoc(), *getTypeConverter());
    ArrayRef<Value>::iterator valueIt = storageValues.begin();
    unsigned storageOffset = 0;
    unsigned offset = 0;
    Value result =
        helper.walkAndExtractVariant(valueIt, storageOffset, offset, valueType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertKGENCustomOpImplsOp
//===----------------------------------------------------------------------===//

struct ConvertKGENOpImpls : ConvertPOPToLLVMPattern<CustomOpImplsOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CustomOpImplsOp op, CustomOpImplsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateKGENToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns,
                                       SymbolTable &symtab,
                                       InterpreterMemoryConverter &imc,
                                       const AttributeIdentifiers &ids) {
  patterns.insert<
      // clang-format off
      ConvertKGENCall,
      ConvertKGENGlobalAddress,
      ConvertKGENOpImpls,
      ConvertKGENStructCreate,
      ConvertKGENStructGEP,
      ConvertKGENStructGet,
      ConvertKGENStructReplace,
      ConvertKGENVariantCreate,
      ConvertKGENVariantGet,
      ConvertKGENVariantIs,
      ConvertKGENRebind,
      ConvertKGENReturn,
      ConvertKGENUnreachable,
      ConvertKGENUndef
      // clang-format on
      >(typeConverter);
  patterns.insert<ConvertKGENFunc>(typeConverter, symtab, ids);
  patterns.insert<ConvertKGENParamConstant, ConvertKGENParamMaterialize>(
      typeConverter, imc);
}

//===----------------------------------------------------------------------===//
// convertGlobals
//===----------------------------------------------------------------------===//

/// Convert all the `kgen.global` operations in the module to LLVM globals. This
/// involves generating `llvm.mlir.global` operations for each but also
/// generating the correct global constructors and destructors.
///
/// In JIT mode, instead of generating `llvm.global_ctors` and
/// `llvm.global_dtors`, an extra pair of constructor and destructor functions
/// are generated with the provided names.
static LogicalResult convertGlobals(ModuleOp module, POPToLLVMTypeConverter &tc,
                                    StringRef globalCtorFnName,
                                    StringRef globalDtorFnName) {
  SmallVector<FlatSymbolRefAttr> ctors, dtors;
  SmallVector<int32_t> priorities;

  for (auto global : llvm::make_early_inc_range(module.getOps<GlobalOp>())) {
    // Replace the `pop.global` with an `llvm.mlir.global`, raise the
    // constructor and destructor into functions, and collect a list of them.
    mlir::IRRewriter b{OpBuilder(global)};
    Type type = tc.convertType(global.getType());
    if (!type)
      return global.emitError("could not convert global type");

    if (global.getCtor()) {
      ctors.push_back(cast<FlatSymbolRefAttr>(global.getCtorAttr()));
      dtors.push_back(cast<FlatSymbolRefAttr>(global.getDtorAttr()));
      priorities.push_back(*global.getPriority());
    }

    // Create the LLVM global.
    bool isExported = global.isExported();
    auto llvmGlobal = b.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, type, /*constant=*/false,
        getLinkageKind(global.getExportKind()), global.getSymName(),
        /*value=*/Attribute());

    // If the global is not exported, then no need to initialize it.
    if (!isExported)
      continue;

    // If the global is exported, explicitly initialize it as undef.
    b.createBlock(&llvmGlobal.getBodyRegion());
    Value undef = b.create<LLVM::UndefOp>(llvmGlobal.getLoc(), type);
    b.create<LLVM::ReturnOp>(llvmGlobal.getLoc(), undef);
  }

  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // HACK: NVPTX doesn't support global destructors.
  if (llvm::is_contained({llvm::Triple::nvptx, llvm::Triple::nvptx64},
                         tc.getTarget().getTriple().getArch()))
    return success();

  auto b = OpBuilder::atBlockBegin(module.getBody());
  // Sort the constructor function indices. Lower priority is earlier.
  SmallVector<unsigned> order =
      llvm::to_vector(llvm::seq<unsigned>(0, priorities.size()));
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    return priorities[lhs] < priorities[rhs];
  });

  auto populateCalls = [&b](LLVM::LLVMFuncOp func,
                            ArrayRef<FlatSymbolRefAttr> refs, auto order) {
    b.createBlock(&func.getRegion());
    for (unsigned i : order)
      b.create<LLVM::CallOp>(func.getLoc(), TypeRange(),
                             cast<FlatSymbolRefAttr>(refs[i]));
    b.create<LLVM::ReturnOp>(func.getLoc(), ValueRange());
  };

  auto type =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()), {});
  LLVM::LLVMFuncOp ctor =
      createLLVMFunc(b, tc.getTarget(), module.getLoc(), globalCtorFnName, type,
                     LLVM::Linkage::Weak);
  LLVM::LLVMFuncOp dtor =
      createLLVMFunc(b, tc.getTarget(), module.getLoc(), globalDtorFnName, type,
                     LLVM::Linkage::Weak);
  populateCalls(ctor, ctors, order);
  populateCalls(dtor, dtors, llvm::reverse(order));

  // Create the `llvm.mlir.global_ctors` and `llvm.mlir.global_dtors`, where
  // each just invokes the respective functions we generated.
  b.setInsertionPointToStart(module.getBody());
  mlir::ArrayAttr prioritiesAttr = b.getArrayAttr(b.getI32IntegerAttr(0));
  b.create<LLVM::GlobalCtorsOp>(
      module.getLoc(),
      b.getArrayAttr(FlatSymbolRefAttr::get(b.getStringAttr(globalCtorFnName))),
      prioritiesAttr);
  b.create<LLVM::GlobalDtorsOp>(
      module.getLoc(),
      b.getArrayAttr(FlatSymbolRefAttr::get(b.getStringAttr(globalDtorFnName))),
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
    Value arrPtr = body->addArgument(LLVM::LLVMPointerType::get(b.getContext()),
                                     b.getLoc());
    return b.create<LLVM::LoadOp>(arrayTy, arrPtr);
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
      body->addArgument(LLVM::LLVMPointerType::get(type.getContext()), loc);
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
  func.setLinkage(LLVM::Linkage::Internal);
  symtab.insert(func);

  // Update the subprogram scope of the wrapped function if it has one, but save
  // the location before it gets changed.
  Location loc = func.getLoc();
  DebugInfo::updateSubprogram(func, newName);

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
class LowerKGENToLLVMPass
    : public KGEN::impl::LowerKGENToLLVMBase<LowerKGENToLLVMPass> {
public:
  using LowerKGENToLLVMBase::LowerKGENToLLVMBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    using LLVM::LLVMDialect;
    auto id = [&](StringRef name) { return StringAttr::get(ctx, name); };

    ids.noalias = id(LLVMDialect::getNoAliasAttrName());
    ids.noundef = id(LLVMDialect::getNoUndefAttrName());
    ids.nonnull = id(LLVMDialect::getNonNullAttrName());

    return success();
  }

  void runOnOperation() override;

private:
  AttributeIdentifiers ids;
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
  target.addLegalOp<KGEN::CallIndirectOp>();
  target.addLegalOp<KGEN::CreateClosureOp>();

  // Collect C exported symbols. The calling convention will have to be
  // rewritten after the lowering.
  SmallVector<StringAttr> exportCFuncs;
  for (auto symbol : theModule.getOps<ExportInterface>())
    if (symbol.isCExported())
      exportCFuncs.push_back(symbol.getLinkageNameAttr());

  // Configure the type converter.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  // HACK HACK HACK: Our current name mangling scheme is not compatible with the
  // NVPTX backend. Change the symbol names to be compatbile.
  if (llvm::is_contained({llvm::Triple::nvptx, llvm::Triple::nvptx64},
                         targetInfo.getTriple().getArch())) {
    DenseMap<StringAttr, StringAttr> renamed;
    for (auto symbol : getOperation().getOps<mlir::SymbolOpInterface>()) {
      StringAttr name = symbol.getNameAttr();
      StringAttr sanitized = sanitizeSymbolToAlnum(symbol.getNameAttr());
      if (name != sanitized) {
        renamed.try_emplace(name, sanitized);
        symbol.setName(sanitized);
      }
    }
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&renamed](FlatSymbolRefAttr symbol) {
      if (auto it = renamed.find(symbol.getAttr()); it != renamed.end())
        return FlatSymbolRefAttr::get(it->second);
      return symbol;
    });
    auto workFn = [](mlir::AttrTypeReplacer &replacer, Operation *op) {
      replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/false);
    };
    std::vector<Operation *> work;
    for (Operation &op : getOperation().getOps())
      work.push_back(&op);
    parallelForEach(&getContext(), work, workFn, replacer);
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
  if (failed(convertGlobals(theModule, typeConverter, globalCtorFnName,
                            globalDtorFnName)))
    return signalPassFailure();

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable &symtab = symtabAnalysis.getTopLevelSymbolTable();
  InterpreterMemoryConverter imc(symtab, typeConverter);
  populateKGENToLLVMPatterns(typeConverter, patterns, symtab, imc, ids);

  DebugInfoTypeConverter debugTypeConverter(typeConverter);
  DebugInfo::populateTypeConversionPatterns(patterns, debugTypeConverter,
                                            typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();

  // Process updates to any exported functions.
  mlir::SymbolUserMap symbolUsers(symtabAnalysis.getSymbolTables(), theModule);
  for (StringAttr sym : exportCFuncs)
    if (auto func = symtab.lookup<LLVM::LLVMFuncOp>(sym))
      processCExportedFunction(func, symbolUsers, symtab, targetInfo);

  // Convert the debug info within the IR.
  debugTypeConverter.applyRecursively(theModule);
}
