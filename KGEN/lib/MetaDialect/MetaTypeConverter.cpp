//===- MetaTypeConverter.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// MetaToLLVMTypeConverter Implementation
//===----------------------------------------------------------------------===//

static Optional<Type> getMLIRTypeForDType(MLIRContext *ctx, DType dtype) {
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

MetaToLLVMTypeConverter::MetaToLLVMTypeConverter(
    mlir::Location loc, const mlir::LowerToLLVMOptions &options)
    : LLVMTypeConverter(loc.getContext(), options), loc(loc) {

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> Optional<Type> {
    auto dtypeConst = type.getDType().template dyn_cast<DTypeConstantAttr>();
    if (!dtypeConst) {
      emitError("dtype not fully specified: ") << type;
      return {};
    }
    return getMLIRTypeForDType(type.getContext(), dtypeConst.getDType());
  };

  // Convert a size expression to a C++ unsigned integer.
  auto convertSize = [&](auto type) -> Optional<unsigned> {
    auto size = type.getSize().template dyn_cast<IntegerAttr>();
    if (!size) {
      emitError("size not fully specified: ") << type;
      return {};
    }
    const APInt &value = size.getValue();
    assert(APInt(value.getBitWidth(), value.getLimitedValue()) == value &&
           "couldn't narrow vector size");
    return value.getLimitedValue();
  };

  // Convert scalar types directly to the dtype.
  addConversion([=](ScalarType scalar) { return convertDType(scalar); });

  // Convert pointer types to bare pointers of the dtype.
  addConversion([=](PointerType pointer) -> Optional<Type> {
    if (Optional<Type> dtype = convertDType(pointer))
      return LLVM::LLVMPointerType::get(*dtype);
    return {};
  });

  // Convert SIMD types to vector types.
  addConversion([=](SIMDType simd) -> Optional<Type> {
    Optional<Type> dtype = convertDType(simd);
    auto size = convertSize(simd);
    if (!dtype || !size)
      return {};
    return mlir::VectorType::get(*size, *dtype);
  });

  // Convert buffers to struct<(i64, ptr<T>)>.
  // TODO: Support unknown dtype and convert fixed-size arrays to pointers.
  addConversion([=](BufferType buffer) -> Optional<Type> {
    Optional<Type> dtype = convertDType(buffer);
    if (!dtype)
      return {};
    return LLVM::LLVMStructType::getLiteral(
        buffer.getContext(), {convertType(IndexType::get(&getContext())),
                              LLVM::LLVMPointerType::get(*dtype)});
  });
}
