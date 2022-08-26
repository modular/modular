//===- LLVMLoweringUtils.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "KGEN/MetaDialect/MetaTypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

Value M::KGEN::emitBufferSizeToLLVM(Location loc, BufferType type, Value buf,
                                    ConversionPatternRewriter &rewriter,
                                    mlir::LLVMTypeConverter &converter) {
  BufferDescriptor buffer(type);
  if (Optional<int64_t> size = buffer.getSize()) {
    return rewriter.create<LLVM::ConstantOp>(loc, converter.getIndexType(),
                                             rewriter.getIndexAttr(*size));
  }
  return rewriter.create<LLVM::ExtractValueOp>(loc, buf,
                                               *buffer.getSizeIndex());
}

Value M::KGEN::emitBufferAddressToLLVM(Location loc, BufferType type, Value buf,
                                       ConversionPatternRewriter &rewriter) {
  BufferDescriptor buffer(type);
  if (buffer.isBarePtr())
    return buf;
  return rewriter.create<LLVM::ExtractValueOp>(loc, buf, *buffer.getPtrIndex());
}

Value M::KGEN::emitBufferDTypeToLLVM(Location loc, BufferType type, Value buf,
                                     ConversionPatternRewriter &rewriter) {
  BufferDescriptor buffer(type);
  if (Optional<DType> dtype = buffer.getDType()) {
    return rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(dtype->getValue()));
  }
  return rewriter.create<LLVM::ExtractValueOp>(loc, buf,
                                               *buffer.getDTypeIndex());
}
