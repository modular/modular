//===- LLVMLoweringUtils.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "KGEN/MetaDialect/MetaTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Value.h"

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace M::KGEN {
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
