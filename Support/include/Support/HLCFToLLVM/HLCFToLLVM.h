//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HLCFTOLLVM_HLCFTOLLVM_H
#define SUPPORT_HLCFTOLLVM_HLCFTOLLVM_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

//===----------------------------------------------------------------------===//
// HLCF Lowering
//===----------------------------------------------------------------------===//

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace M::HLCF {
/// Lower all control-flow trees contained within the provided operation to
/// LLVM.
LogicalResult lowerControlFlowToLLVM(Operation *op,
                                     mlir::LLVMTypeConverter &typeConverter);

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_LOWERHLCFTOLLVMPASS
#include "Support/HLCFToLLVM/HLCFToLLVM.h.inc"
} // namespace M::HLCF

#endif // SUPPORT_HLCFTOLLVM_HLCFTOLLVM_H
