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

Optional<Type> M::KGEN::getMLIRTypeForDType(MLIRContext *ctx, KGENDType dtype) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);

  if (dtype.isAddress())
    return LLVM::LLVMPointerType::get(ctx);

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

Type M::KGEN::getLLVMPointerTo(MLIRContext *ctx, KGENDType dtype) {
  if (Optional<Type> type = getMLIRTypeForDType(ctx, dtype))
    return LLVM::LLVMPointerType::get(*type);
  return LLVM::LLVMPointerType::get(ctx);
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(
    mlir::Location loc, const mlir::LowerToLLVMOptions &options)
    : LLVMTypeConverter(loc.getContext(), options), loc(loc) {

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> Optional<Type> {
    if (Optional<KGENDType> dtype = type.getResolvedDType())
      return getMLIRTypeForDType(type.getContext(), *dtype);
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
