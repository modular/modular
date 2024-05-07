//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::DebugInfo {
//===----------------------------------------------------------------------===//
// TargetAdapter
//===----------------------------------------------------------------------===//
struct TargetAdapter {
  /// Conversion patterns.
  std::function<void(DIAttrTypeReplacer &, RewritePatternSet &)>
      populateConversionPatterns;

  /// Custom massaging.
  using DebugAdapterFn = std::function<void(mlir::ModuleOp)>;
  DebugAdapterFn preTranslationAdapter;
  DebugAdapterFn postTranslationAdapter;
};

/// Get the corresponding adapter for the target.
TargetAdapter getTargetAdapter(M::TargetInfoAttr target);

/// Default adapter for targets without a more specific adapter.
TargetAdapter getFallbackAdapter();

//===----------------------------------------------------------------------===//
// Common Routines
//===----------------------------------------------------------------------===//
void populateFallbackConversionPatterns(DIAttrTypeReplacer &replacer,
                                        RewritePatternSet &patterns);

/// Sink kill Debug Value ops so that they are the last instructions from
/// their source line. This way variables are guaranteed to be killed only at
/// the end of the line.
void sinkDebugKills(mlir::Operation *op);

/// This function converts instances of llvm.dbg.value to llvm.dbg.declare when
/// desirable. LLVM optimizations and codegen often muck up the use of
/// llvm.dbg.value (and other debug intrinsics), which creates subpar debugging
/// experiences. Converting to llvm.dbg.declare provides a more stable debugging
/// environment, and more closely matches what a traditional frontend would
/// provide in O0 modes.
///
/// The current conversion policy considers two separate axes:
/// - The number of dbg.values for a variable (regardless of whether the value
/// is undef or not) determines whether dbg.value or dbg.declare is used.
/// - The number of non-undef dbg.values for a variable determines whether we
/// allocate to stack or not.
///
/// Or, listing out the possible combinations:
/// - =1 dbg.value: use dbg.declare, allocate to stack (i.e. var is alive for
/// its entire scope)
/// - >1 dbg.value, =1 non-undef: use dbg.value, allocate to stack (i.e. var is
/// stationary for its entire lifetime)
/// - >1 dbg.value, >1 non-undef: use dbg.declare, do allocate to stack but only
/// for debuginfo, don't replace original SSA variable reads with stack reads
/// (in case of re-ordering) (i.e. var moves around, or exists as fragments).
///
/// TODO: As we grow support we may want to consider making this optional
/// depending on the debug mode.
void convertDbgValueToDeclare(mlir::ModuleOp module);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H
