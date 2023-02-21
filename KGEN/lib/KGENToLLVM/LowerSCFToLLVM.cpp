//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSCFTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSCFToLLVMPass
    : public KGEN::impl::LowerSCFToLLVMBase<LowerSCFToLLVMPass> {
  using LowerSCFToLLVMBase::LowerSCFToLLVMBase;

  void runOnOperation() override;
};
} // namespace

void LowerSCFToLLVMPass::runOnOperation() {
  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(getOperation());
  if (!targetInfo) {
    mlir::emitError(getOperation()->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Run HLCF lowerings.
  if (failed(HLCF::lowerControlFlowToLLVM(
          getOperation(), getAnalysis<HLCF::ControlFlowTreeAnalysis>(),
          typeConverter)))
    return signalPassFailure();

  // Erase unreachable blocks that might arise during HLCF lowering.
  mlir::IRRewriter rewriter(&getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter, getOperation()->getRegions());
}
