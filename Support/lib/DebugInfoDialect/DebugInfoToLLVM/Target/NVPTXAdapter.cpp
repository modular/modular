//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "NVPTXAdapter.h"

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace M::DebugInfo;

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Convert LineTableLocOp into a "nop" equivalent in PTX.
struct ConvertLineTableLocOp : public OpRewritePattern<LineTableLocOp> {
  ConvertLineTableLocOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<LineTableLocOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(LineTableLocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<LLVM::InlineAsmOp>(
        replacer.replace<LocationAttr>(op.getLoc()), TypeRange{}, ValueRange{},
        "pmevent.mask 0;", "", /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr());
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

static void populateNVPTXConversionPatterns(DIAttrTypeReplacer &replacer,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertLineTableLocOp>(patterns.getContext(), replacer);
}

//===----------------------------------------------------------------------===//
// Custom Massaging
//===----------------------------------------------------------------------===//

/// NVPTX does not support variables that have more than one location. This
/// means we cannot have a variable that has limited lifetime. Remove KillOps so
/// that instead of emitting multiple llvm DbgValueOps, we just emit a single
/// DbgDeclareOp when the variable does not change location.
static void removeDebugKills(mlir::ModuleOp module) {
  module->walk([](DebugInfo::KillOp kill) { kill->erase(); });
}

//===----------------------------------------------------------------------===//
// getNVPTXAdapter
//===----------------------------------------------------------------------===//

TargetAdapter DebugInfo::getNVPTXAdapter() {
  return TargetAdapter{populateNVPTXConversionPatterns, removeDebugKills,
                       convertDbgValueToDeclare};
}
