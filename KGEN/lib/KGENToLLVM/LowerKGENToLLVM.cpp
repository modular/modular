//===- LowerKGENToLLVM.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/ForwardDecls.h"
#include "Support/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void M::KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                       const LowerToLLVMOptions &options) {
  // Run the canonicalizer before the lowering passes.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  // FIXME: We don't necessarily always want to emit opaque wrappers. Split this
  //        code up better because there's 2 semi-separate compilation models
  //        here.
  SmallVector<StringRef> breakUpStructs;
  if (options.topLevelKernel.hasValue())
    breakUpStructs.push_back(options.topLevelKernel);
  pm.addPass(createConvertKGENToLLVMPass(breakUpStructs, {},
                                         options.emitOpaqueWrappers));

  // Run all LLVM lowering passes.
  pm.addPass(createLowerGlobalPOPToLLVM());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(createConvertPOPToLLVMPass());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(createConvertSCFToLLVMPass());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(index::createIndexToLLVM());

  // And finally canonicalize again.
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());
}

void M::KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}
