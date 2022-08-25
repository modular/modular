//===- MetaTypeConverter.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// BufferDescriptor
//===----------------------------------------------------------------------===//

BufferDescriptor::BufferDescriptor(BufferType buffer)
    : buffer(buffer), size(buffer.getSize().dyn_cast_or_null<IntegerAttr>()),
      dtype(buffer.getDType().dyn_cast_or_null<DTypeConstantAttr>()) {}

bool BufferDescriptor::isBarePtr() const { return size && dtype; }

Optional<int64_t> BufferDescriptor::getSizeIndex() const {
  if (size)
    return None;
  // The size field is always first, if present.
  return 0;
}

Optional<int64_t> BufferDescriptor::getDTypeIndex() const {
  if (dtype)
    return None;
  // The size field is offset by the size field, if present, and follows the
  // pointer field.
  return size ? 1 : 2;
}

Optional<int64_t> BufferDescriptor::getPtrIndex() const {
  if (isBarePtr())
    return None;
  // The pointer field is offset by the size field.
  return size ? 0 : 1;
}

Optional<int64_t> BufferDescriptor::getSize() const {
  if (!size)
    return None;
  return size.getInt();
}

Optional<DType> BufferDescriptor::getDType() const {
  if (!dtype)
    return None;
  return dtype.getDType();
}

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
    switch (dtype.getValue()) {
    default:
      break;
    case DType::f16:
      return FloatType::getF16(ctx);
    case DType::bf16:
      return FloatType::getBF16(ctx);
    case DType::f32:
      return FloatType::getF32(ctx);
    case DType::f64:
      return FloatType::getF64(ctx);
    }
  }

  return {};
}

Type M::KGEN::getLLVMPointerTo(MLIRContext *ctx, DType dtype) {
  if (Optional<Type> type = getMLIRTypeForDType(ctx, dtype))
    return LLVM::LLVMPointerType::get(*type);
  return LLVM::LLVMPointerType::get(ctx);
}

/// Convert a buffer type to an LLVM type. Both the element type and size can be
/// unknown.
///
/// - If both the element type and size are known, then the buffer is lowered to
///   a raw pointer.
/// - If the size is unknown, the buffer is lowered to a struct with an
///   `index`-typed size field.
/// - The element type is unknown, the buffer is lowered to a struct with an
///   `i8`-typed discriminant field with the value of `DType::getValue` and the
///   pointer becomes untyped.
static Type convertBufferType(mlir::LLVMTypeConverter &converter,
                              Optional<Type> dtype, Optional<uint64_t> size) {
  MLIRContext *ctx = &converter.getContext();
  SmallVector<Type, 3> fields;
  if (!size)
    fields.push_back(converter.getIndexType());

  if (!dtype)
    fields.append({LLVM::LLVMPointerType::get(ctx), Builder(ctx).getI8Type()});
  else
    fields.push_back(LLVM::LLVMPointerType::get(*dtype));

  return fields.size() == 1 ? fields.front()
                            : LLVM::LLVMStructType::getLiteral(ctx, fields);
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

  // Convert pointer types to bare pointers of the dtype. If the dtype is
  // unspecified, return an untyped pointer.
  addConversion([=](PointerType pointer) -> Optional<Type> {
    return getLLVMPointerTo(&getContext(), pointer.resolveDType());
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

  // Convert buffer types to pointers or structs.
  addConversion([=](BufferType buffer) -> Optional<Type> {
    return convertBufferType(*this, convertDType(buffer), convertSize(buffer));
  });

  // Convert data type types to `i8`.
  addConversion([=](DTypeType dtype) -> Optional<Type> {
    return Builder(&getContext()).getI8Type();
  });
}
