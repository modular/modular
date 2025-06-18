//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/IR/Verifier.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_KGENVERIFIERPASS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

/// Verify correctness of the attribute for a given target.
/// * Fail if MemoryBlob attribute on GPU will lowered into heap
///   allocation, which is known not to generally work.
static LogicalResult verifyAttributes(Operation *op, TargetInfoAttr target) {
  mlir::AttrTypeReplacer replacer;
  bool hasHeapAllocation = false;
  replacer.addReplacement(
      [&](MemorySpaceAttr space) -> std::pair<Attribute, WalkResult> {
        for (MemoryBlobAttr blob : space) {
          if (isGlobalBlob(blob))
            continue;
          if (blob.getKind() != MemoryKind::Stack) {
            hasHeapAllocation = true;
            return {nullptr, WalkResult::interrupt()};
          }
        }
        return {nullptr, WalkResult::advance()};
      });

  replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                             /*replaceLocs=*/false,
                             /*replaceTypes=*/false);

  if (hasHeapAllocation && isGPUTriple(target.getTriple())) {
    op->emitError("heap allocation is not supported on GPU");
    return failure();
  }
  return success();
}

/// Verify operation correctness for a target.
/// * Fail if heap allocation or deallocation is encountered on GPU
/// * Fail if its attribute is not legal for the target (see verifyAttributes)
static LogicalResult verifyOperation(Operation *op, TargetInfoAttr target) {
  // Cannot move easily verification of these ops inside their `verify`
  // functions, since GPU module may have lots of these operations from the
  // beginning, but they may not be used and will be cleared by some
  // optimization later.
  if (isa<POP::AlignedAllocOp>(op)) {
    op->emitError("heap allocation is not supported on GPU");
    return failure();
  }
  if (isa<POP::AlignedFreeOp>(op)) {
    op->emitError("heap deallocation is not supported on GPU");
    return failure();
  }
  return verifyAttributes(op, target);
}

namespace {
struct KGENVerifierPass : public impl::KGENVerifierPassBase<KGENVerifierPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(mlir::verify(op)))
      signalPassFailure();

    TargetInfoAttr target = lookupTargetInfo(op);
    if (!target || !isGPUTriple(target.getTriple()))
      return;

    // Number of errors encountered during verification.
    size_t numErrors = 0;

    // Following code is GPU-specific verification that makes sure no operation
    // introduces heap allocation for GPU compilation.
    if (op->walk([&](Operation *operation) {
            if (failed(verifyOperation(operation, target)))
              ++numErrors;
            if (numErrors >= *KGENPassCLOptions::kgenVerifierMaxErrors())
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
