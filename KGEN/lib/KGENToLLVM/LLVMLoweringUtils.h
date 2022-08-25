//===- LLVMLoweringUtils.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
Value emitBufferAddressToLLVM(Location loc, Value buf, Value adaptorBuf,
                              ConversionPatternRewriter &rewriter);
}

#endif // KGEN_LLVM_LOWERING_UTILS_H
