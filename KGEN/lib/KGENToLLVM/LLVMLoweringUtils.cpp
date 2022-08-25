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

Value M::KGEN::emitBufferAddressToLLVM(Location loc, Value buf,
                                       Value adaptorBuf,
                                       ConversionPatternRewriter &rewriter) {
  BufferDescriptor buffer(buf.getType().cast<BufferType>());
  if (buffer.isBarePtr())
    return adaptorBuf;
  return rewriter.create<LLVM::ExtractValueOp>(loc, adaptorBuf,
                                               *buffer.getPtrIndex());
}
