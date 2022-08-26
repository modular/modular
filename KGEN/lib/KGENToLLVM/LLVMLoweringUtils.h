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
  /// Returns the known dtype of the buffer, if it has one.
  llvm::Optional<DType> getDType() const;

private:
  /// The buffer type.
  BufferType buffer;
  /// An optional known size of the buffer.
  mlir::IntegerAttr size;
  /// An optional known dtype of the buffer.
  DTypeConstantAttr dtype;
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

//===----------------------------------------------------------------------===//
// LLVM Code Emitters
//===----------------------------------------------------------------------===//

/// Emit the LLVM code to get the size of a buffer. Return the size value.
Value emitBufferSizeToLLVM(Location loc, BufferType type, Value buf,
                           ConversionPatternRewriter &rewriter,
                           mlir::LLVMTypeConverter &converter);
inline Value emitBufferSizeToLLVM(Location loc, Value buf, Value adaptorBuf,
                                  ConversionPatternRewriter &rewriter,
                                  mlir::LLVMTypeConverter &converter) {
  return emitBufferSizeToLLVM(loc, buf.getType().cast<BufferType>(), adaptorBuf,
                              rewriter, converter);
}
/// Emit the LLVM code to get the address of a buffer. Return the address value.
Value emitBufferAddressToLLVM(Location loc, BufferType type, Value buf,
                              ConversionPatternRewriter &rewriter);
inline Value emitBufferAddressToLLVM(Location loc, Value buf, Value adaptorBuf,
                                     ConversionPatternRewriter &rewriter) {
  return emitBufferAddressToLLVM(loc, buf.getType().cast<BufferType>(),
                                 adaptorBuf, rewriter);
}
/// Emit the LLVM code to get the dtype of a buffer. Return the dtype value.
Value emitBufferDTypeToLLVM(Location loc, BufferType type, Value buf,
                            ConversionPatternRewriter &rewriter);
inline Value emitBufferDTypeToLLVM(Location loc, Value buf, Value adaptorBuf,
                                   ConversionPatternRewriter &rewriter) {
  return emitBufferDTypeToLLVM(loc, buf.getType().cast<BufferType>(),
                               adaptorBuf, rewriter);
}

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
