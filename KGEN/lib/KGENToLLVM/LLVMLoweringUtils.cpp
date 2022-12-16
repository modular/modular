//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

Optional<Type> M::KGEN::getMLIRTypeForDType(MLIRContext *ctx, KGENDType dtype,
                                            size_t indexBitwidth) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);

  if (dtype.isAddress())
    return LLVM::LLVMPointerType::get(ctx);

  if (dtype.isIndex())
    return IntegerType::get(ctx, indexBitwidth);

  // This intentionally discards signed-ness because LLVM is signless.
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getIntegerWidthInBits());

  if (dtype.isFloat()) {
    if (FloatType fpType = getEquivalentFloatType(ctx, dtype))
      return fpType;
    return {};
  }

  return {};
}

Type M::KGEN::getLLVMPointerTo(MLIRContext *ctx, KGENDType dtype,
                               size_t indexBitwidth) {
  if (Optional<Type> type = getMLIRTypeForDType(ctx, dtype, indexBitwidth))
    return LLVM::LLVMPointerType::get(*type);
  return LLVM::LLVMPointerType::get(ctx);
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(
    mlir::Location loc, const mlir::LowerToLLVMOptions &options)
    : LLVMTypeConverter(loc.getContext(), options), loc(loc) {

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> Optional<Type> {
    if (Optional<KGENDType> dtype = type.getResolvedDType())
      return getMLIRTypeForDType(type.getContext(), *dtype,
                                 options.getIndexBitwidth());
    return {};
  };

  // Convert a size expression to a C++ unsigned integer.
  auto convertSize = [&](auto type) -> Optional<uint64_t> {
    auto size = dyn_cast_if_present<IntegerAttr>(type.getSize());
    if (!size)
      return {};
    const APInt &value = size.getValue();
    assert(APInt(value.getBitWidth(), value.getLimitedValue()) == value &&
           "couldn't narrow vector size");
    return value.getLimitedValue();
  };

  // Convert pointer types to LLVM pointer types. If the element type is
  // unspecified, return an opaque pointer.
  addConversion([=](POP::PointerType pointer) -> Optional<Type> {
    if (Type type = pointer.getResolvedElementType())
      if (Type elementType = convertType(type))
        return LLVM::LLVMPointerType::get(elementType);
    return LLVM::LLVMPointerType::get(pointer.getContext());
  });

  // Convert array types to LLVM array types.
  addConversion([=](POP::ArrayType array) -> Optional<Type> {
    Optional<int64_t> size = array.getResolvedSize();
    Type elementType = array.getResolvedElementType();
    if (!size || !elementType)
      return {};
    elementType = convertType(elementType);
    if (!elementType)
      return {};
    return LLVM::LLVMArrayType::get(elementType, *size);
  });

  // Convert struct types to LLVM literal structs.
  addConversion([=](POP::StructType structType) -> Optional<Type> {
    SmallVector<Type> elementTypes;
    elementTypes.reserve(structType.getNumElements());
    for (TypedAttr elementType : structType.getElementTypes()) {
      auto typeCst = dyn_cast<ConcreteTypeConstantAttr>(elementType);
      if (!typeCst)
        return {};
      Type converted = convertType(typeCst.getValue());
      if (!converted)
        return {};
      elementTypes.push_back(converted);
    }
    return LLVM::LLVMStructType::getLiteral(&getContext(), elementTypes);
  });

  // Convert SIMD types to vector types.
  addConversion([=](POP::SIMDType simd) -> Optional<Type> {
    Optional<Type> dtype = convertDType(simd);
    Optional<uint64_t> size = convertSize(simd);
    if (!dtype)
      return {};
    if (!size) {
      emitError("SIMD size not fully specified: ") << simd;
      return {};
    }

    // Scalar case, size = 1
    if (*size == 1)
      return *dtype;

    // Vector case, size != 1
    return LLVM::getFixedVectorType(*dtype, *size);
  });

  // Convert data type types to `i8`.
  addConversion([=](DTypeType dtype) -> Optional<Type> {
    return Builder(&getContext()).getI8Type();
  });

  // Convert variant types to a struct with enough space to contain the largest
  // variant type plus a discriminator.
  addConversion([=](POP::VariantType variant) -> Optional<Type> {
    // TODO: The generated assembly is sensitive to the content type of the
    // variant type. This needs to be optimized. For now, use an array of
    // word-size integers.
    uint64_t maxSize = 0;
    for (TypedAttr typeExpr : variant.getTypes()) {
      Type variantType = typeExpr.cast<ConcreteTypeConstantAttr>().getValue();
      Type type = convertType(variantType);
      if (!type)
        return {};
      maxSize = std::max(maxSize, llvm::alignTo(dl.getTypeSize(type),
                                                dl.getTypeABIAlignment(type)));
    }
    auto contentType = LLVM::LLVMArrayType::get(
        getIndexType(),
        llvm::divideCeil(maxSize * CHAR_BIT, getIndexTypeBitwidth()));
    // Compute the smallest integer to contain the discriminator. If there is
    // only one variant type, use i1 since LLVM does not accept i0.
    auto discrType = IntegerType::get(
        &getContext(),
        std::max(1u, llvm::Log2_32_Ceil(variant.getTypes().size())));
    return LLVM::LLVMStructType::getLiteral(&getContext(),
                                            {contentType, discrType});
  });
}

//===----------------------------------------------------------------------===//
// POPToLLVMDebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// Build an integer or floating point debug type `T` with the given name and
/// width.
template <typename T>
auto buildIntFpDebugType(MLIRContext *ctx, StringRef name, unsigned width,
                         unsigned conservativeAlign) {
  // TODO: This should be driven by target info in the longer term.
  uint32_t align =
      llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT)) * CHAR_BIT;
  align = std::min(align, conservativeAlign);

  uint64_t size = llvm::alignTo(width, align);
  return T::get(ctx, name, size, align);
}

static DebugInfo::DIType
buildDebugTypeFromDType(MLIRContext *ctx, uint8_t dtype, unsigned indexWidth) {
  // Process various builtin dtypes.
  switch (dtype) {
  case DType::kBool:
    return buildIntFpDebugType<DebugInfo::DIBasicBoolType>(ctx, "bool", 8, 8);
  case DType::si1:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si1", 1, 8);
  case DType::ui1:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui1", 1, 8);
  case DType::si2:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si2", 2, 8);
  case DType::ui2:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui2", 2, 8);
  case DType::si4:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si4", 4, 8);
  case DType::ui4:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui4", 4, 8);
  case DType::si8:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si8", 8, 8);
  case DType::ui8:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui8", 8, 8);
  case DType::si16:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si16", 16, 16);
  case DType::ui16:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui16", 16, 16);
  case DType::si32:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si32", 32, 32);
  case DType::ui32:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui32", 32, 32);
  case DType::si64:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si64", 64, 64);
  case DType::ui64:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui64", 64, 64);
  case DType::si128:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si128", 128,
                                                           64);
  case DType::ui128:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui128", 128,
                                                           64);

  case DType::f8:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f8", 8, 8);
  case DType::f16:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f16", 16, 16);
  case DType::f32:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f32", 32, 32);
  case DType::f64:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f64", 64, 64);
  case DType::f128:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f128", 128,
                                                            64);
  case DType::bf16:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "bf16", 16,
                                                            16);
  case DType::f24:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f24", 24, 32);
  case DType::f80:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f80", 80, 64);
  case DType::tf32:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "tf32", 32,
                                                            32);

  case KGENDType::index:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(
        ctx, "index", indexWidth, indexWidth);

    // TODO: Process the remaining dtypes.
  default:
    return nullptr;
  }
}

POPToLLVMDebugInfoTypeConverter::POPToLLVMDebugInfoTypeConverter(
    POPToLLVMTypeConverter &converter) {
  // Let the LLVM conversion handle a majority of the debug info generation.
  addUnresolvedConverter(converter);

  // Add direct debug info conversions.
  addConversion([&](POP::SIMDType type) -> Optional<Type> {
    // We can only build debug info if the dtype and size have been resolved.
    Optional<KGENDType> dtype = type.getResolvedDType();
    Optional<int64_t> size = type.getResolvedSize();
    if (!dtype || !size)
      return std::nullopt;

    // Get the base debug type from the dtype.
    DebugInfo::DIType baseType = buildDebugTypeFromDType(
        type.getContext(), dtype->getValue(), converter.getIndexTypeBitwidth());
    if (!baseType)
      return std::nullopt;

    // Single element SIMD becomes a scalar, multi-element become vectors.
    if (*size == 1)
      return baseType;
    return DebugInfo::DIVectorType::get(baseType, *size);
  });

  // TODO: Add debug generation for variant and dtype.
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

Value KGEN::createAllocaAtEntry(Operation *op, Type type,
                                PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(
      &op->getParentOfType<mlir::FunctionOpInterface>()
           .getFunctionBody()
           .front());
  Value one = rewriter.create<LLVM::ConstantOp>(op->getLoc(),
                                                rewriter.getI64IntegerAttr(1));
  return rewriter.create<LLVM::AllocaOp>(op->getLoc(),
                                         LLVM::LLVMPointerType::get(type), one);
}

/// Compute the bytecount of a buffer of numElements with specified elementType.
int64_t KGEN::getByteCount(Type elementType, IntegerAttr numElements) {
  MLIRContext *ctx = elementType.getContext();
  auto target = KGEN::TargetInfoAttr::getForHost(ctx);

  // Return the element type size multiplied by the size.
  if (auto arry = llvm::dyn_cast<LLVM::LLVMArrayType>(elementType)) {
    Type arrayElementType = arry.getElementType();
    return llvm::alignTo(KGEN::getByteCount(arrayElementType),
                         *DataLayoutInterface::getTypeAlignInBytes(
                             target, arrayElementType)) *
           arry.getNumElements();
  }

  Optional<int64_t> elementByteSize =
      DataLayoutInterface::getTypeSizeInBytes(target, elementType);
  assert(elementByteSize.has_value() && "elementByteSize must be resolved");
  if (numElements)
    return *elementByteSize * numElements.getInt();
  return *elementByteSize;
}
