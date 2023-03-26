//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFTOLLVM_H
#define KGEN_HLCFTOLLVM_H

#include "KGEN/HLCFDialect/Analysis/ControlFlowTree.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

//===----------------------------------------------------------------------===//
// HLCF Lowering
//===----------------------------------------------------------------------===//

namespace mlir {
class AnalysisManager;
class LLVMTypeConverter;
class RewriterBase;
} // namespace mlir

namespace M::HLCF {
/// Lower all control-flow trees contained within the provided operation to
/// LLVM, given the top-level analysis manager.
LogicalResult lowerControlFlowToLLVM(Operation *op,
                                     ControlFlowTreeAnalysis &analysis,
                                     mlir::LLVMTypeConverter &typeConverter);

/// Lower a return-like operation to LLVM, packing the results if necessary.
LogicalResult
lowerReturnOperationToLLVM(Operation *op, ValueRange operands,
                           mlir::RewriterBase &rewriter,
                           mlir::LLVMTypeConverter &typeConverter);
} // namespace M::HLCF

#endif // KGEN_HLCFTOLLVM_H
