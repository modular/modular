//===- LLVMLoweringUtils.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// MetaToLLVMTypeConverter
//===----------------------------------------------------------------------===//

Optional<Type> M::KGEN::getMLIRTypeForDType(MLIRContext *ctx, DType dtype) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);
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

Type M::KGEN::getLLVMPointerTo(MLIRContext *ctx, DType dtype) {
  if (Optional<Type> type = getMLIRTypeForDType(ctx, dtype))
    return LLVM::LLVMPointerType::get(*type);
  return LLVM::LLVMPointerType::get(ctx);
}

MetaToLLVMTypeConverter::MetaToLLVMTypeConverter(
    mlir::Location loc, const mlir::LowerToLLVMOptions &options)
    : LLVMTypeConverter(loc.getContext(), options), loc(loc) {

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> Optional<Type> {
    if (DType dtype = type.resolveDType(); !dtype.isInvalid())
      return getMLIRTypeForDType(type.getContext(), dtype);
    return {};
  };

  // Convert a size expression to a C++ unsigned integer.
  auto convertSize = [&](auto type) -> Optional<uint64_t> {
    auto size = type.getSize().template dyn_cast_or_null<IntegerAttr>();
    if (!size)
      return {};
    const APInt &value = size.getValue();
    assert(APInt(value.getBitWidth(), value.getLimitedValue()) == value &&
           "couldn't narrow vector size");
    return value.getLimitedValue();
  };

  // Convert scalar types directly to the dtype.
  addConversion([=](ScalarType scalar) {
    Optional<Type> dtype = convertDType(scalar);
    if (!dtype)
      emitError("scalar dtype not fully specified: ") << scalar;
    return dtype;
  });

  // Convert pointer types to LLVM pointer types. If the element type is
  // unspecified, return an opaque pointer.
  addConversion([=](PointerType pointer) -> Optional<Type> {
    Type type = pointer.resolveElementType();
    if (!type)
      return LLVM::LLVMPointerType::get(pointer.getContext());
    if (Type elementType = convertType(type))
      return LLVM::LLVMPointerType::get(elementType);
    return {};
  });

  // Convert array types to LLVM array types.
  addConversion([=](POP::ArrayType array) -> Optional<Type> {
    Optional<int64_t> size = array.resolveSize();
    Type elementType = array.resolveElementType();
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
      auto typeCst = elementType.dyn_cast<ConcreteTypeConstantAttr>();
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
  addConversion([=](SIMDType simd) -> Optional<Type> {
    Optional<Type> dtype = convertDType(simd);
    Optional<uint64_t> size = convertSize(simd);
    if (dtype && size)
      return LLVM::getFixedVectorType(*dtype, *size);

    // Emit an error.
    if (!dtype)
      emitError("SIMD dtype not fully specified: ") << simd;
    if (!size)
      emitError("SIMD size not fully specified: ") << simd;
    return {};
  });

  // Convert data type types to `i8`.
  addConversion([=](DTypeType dtype) -> Optional<Type> {
    return Builder(&getContext()).getI8Type();
  });
}
