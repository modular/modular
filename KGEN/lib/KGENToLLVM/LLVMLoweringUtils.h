//===- LLVMLoweringUtils.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "KGEN/MetaDialect/MetaTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Value.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// BufferDescriptor
//===----------------------------------------------------------------------===//

/// Buffer types can be lowered to one of four possible LLVM types depending on
/// whether their sizes are known or their dtypes are known. These are, for
/// example,
///
/// - `!meta.buffer<2, f32>` -> `!llvm.ptr<f32>`
/// - `!meta.buffer<?, f32>` -> `!llvm.struct<(index, ptr<f32>)>`
/// - `!meta.buffer<2, ?>` -> `!llvm.struct<(ptr<i8>, i8)>`
/// - `!meta.buffer<?, ?>` -> `!llvm.struct<(index, ptr<i8>, i8)>`
///
/// This class simplifies interactions with buffers lowered to LLVM by computing
/// the struct field indices.
class BufferDescriptor {
public:
  BufferDescriptor(BufferType buffer);

  /// Returns true if the buffer is converted to a bare pointer.
  bool isBarePtr() const;
  /// Returns the index of the size field, if the buffer needs one.
  llvm::Optional<int64_t> getSizeIndex() const;
  /// Returns the index of the dtype field, if the buffer needs one.
  llvm::Optional<int64_t> getDTypeIndex() const;
  /// Returns the index of the pointer field, if the buffer is not a bare
  /// pointer.
  llvm::Optional<int64_t> getPtrIndex() const;
  /// Returns the known size of the buffer, if it has one.
  llvm::Optional<int64_t> getSize() const;
  /// Returns the known dtype of the buffer or invalid if it lacks one.
  DType getDType() const;

  /// Get the buffer type.
  BufferType getType() { return buffer; }

private:
  /// The buffer type.
  BufferType buffer;
  /// An optional known size of the buffer.
  mlir::IntegerAttr size;
  /// An optional known dtype of the buffer.
  DTypeConstantAttr dtype;
};

//===----------------------------------------------------------------------===//
// BufferDescriptorBuilder
//===----------------------------------------------------------------------===//

/// This helper class generates the LLVM code to interact with concrete buffers.
class BufferDescriptorBuilder : public BufferDescriptor {
public:
  /// Create a builder given the original buffer value.
  BufferDescriptorBuilder(Value buf, Location loc, OpBuilder &builder,
                          mlir::LLVMTypeConverter &converter)
      : BufferDescriptorBuilder(buf.getType().cast<BufferType>(), loc, builder,
                                converter) {}
  /// Create a builder given the buffer type.
  BufferDescriptorBuilder(BufferType type, Location loc, OpBuilder &builder,
                          mlir::LLVMTypeConverter &converter)
      : BufferDescriptor(type), loc(loc), builder(builder),
        converter(converter) {}

  /// Emit the code to get the size of the buffer.
  Value emitGetSize(Value buf);
  /// Emit the code to get the dtype of the buffer.
  Value emitGetDType(Value buf);
  /// Emit the code to get the address of the buffer.
  Value emitGetPtr(Value buf);
  /// Emit the code to set the size of the buffer.
  Value emitSetSize(Value buf, Value size);
  /// Emit the code to set the dtype of the buffer.
  Value emitSetDType(Value buf, Value dtype);
  /// Emit the code to set the address of the buffer.
  Value emitSetPtr(Value buf, Value addr);

  /// Get an empty buffer.
  Value emitUndef();

private:
  /// The location to use when building ops.
  Location loc;
  /// The op builder to use.
  OpBuilder &builder;
  /// The LLVM type converter to use.
  mlir::LLVMTypeConverter &converter;
};

//===----------------------------------------------------------------------===//
// MetaToLLVMTypeConverter
//===----------------------------------------------------------------------===//

/// Get the MLIR type for a data type.
llvm::Optional<mlir::Type> getMLIRTypeForDType(mlir::MLIRContext *ctx,
                                               DType dtype);

/// Get an LLVM pointer to the given dtype. If the dtype is unknown, return an
/// untyped pointer.
mlir::Type getLLVMPointerTo(mlir::MLIRContext *ctx, DType dtype);

/// This type converter maps fully-specified meta dialect parametric types and
/// built-in MLIR types to LLVM types.
class MetaToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  MetaToLLVMTypeConverter(mlir::Location loc,
                          const mlir::LowerToLLVMOptions &options);

  /// Report an error or conversion failure.
  /// TODO: TypeConverter needs an error reporting mechanism.
  mlir::InFlightDiagnostic emitError(llvm::StringRef msg) {
    return mlir::emitError(loc) << msg;
  }

private:
  /// A location used to report conversion failures.
  mlir::Location loc;
};

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
