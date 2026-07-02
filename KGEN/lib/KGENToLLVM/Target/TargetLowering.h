//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Per-target policy for the MLIR-level KGEN/POP -> LLVM-dialect lowering. This
// is the lowering-stage counterpart to ObjectCompiler's `TargetBackend` (the
// codegen-stage abstraction): it lives in KGENToLLVM because the lowering
// passes here are below the codegen layer and cannot depend on `TargetBackend`.
// Both are dispatched by triple, so a target is described once per stage.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H
#define KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H

#include "Support/MDialect/MAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace llvm {
class Triple;
template <class C>
struct object_creator;
} // namespace llvm

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace M::KGEN {

/// Per-target policy for the KGEN/POP -> LLVM-dialect lowering, dispatched by
/// triple via `TargetLoweringRegistry`. A target contributes its own rewrite
/// patterns (the same way a compiler plugin already does); the default
/// contributes none (host-like).
///
/// As the inline per-target branches in the lowering passes are migrated, the
/// per-target logic moves into target-owned patterns registered through these
/// hooks, and the passes just call
/// `TargetLoweringRegistry::lookup(triple)->populate...`.
class TargetLowering {
public:
  virtual ~TargetLowering() = default;

  /// Short name of the target lowering, used for diagnostics.
  virtual llvm::StringRef name() const = 0;
  /// Whether this lowering handles `triple`.
  virtual bool matches(const llvm::Triple &triple) const = 0;

  /// Resolves `triple` to the concrete lowering that should handle it, or null
  /// if this one does not. The default returns `this` when `matches`; a
  /// lowering that owns nested lowerings (e.g. one discovered at runtime) can
  /// override this to return one of them. The registry prefers a resolution
  /// that differs from the lowering it queried, so a nested lowering takes
  /// precedence over a direct match.
  virtual const TargetLowering *resolve(const llvm::Triple &triple) const {
    return matches(triple) ? this : nullptr;
  }

  /// Contributes this target's patterns to the POP -> LLVM lowering
  /// (LowerPOPToLLVM): e.g. fp8 / bf16 casts that lower to ISA intrinsics. The
  /// default contributes none. Mirrors the plugin's
  /// `populateLowerPOPToLLVMPatterns`, so a plugin is just another target
  /// lowering. Called after the generic patterns; target patterns use a higher
  /// benefit so they win for the ops they handle.
  virtual void
  populateLowerPOPToLLVMPatterns(mlir::RewritePatternSet &patterns,
                                 mlir::LLVMTypeConverter &converter,
                                 TargetInfoAttr target) const {}
};

/// Registry of `TargetLowering`s, dispatched by triple. Lowering-stage
/// counterpart to `TargetBackendRegistry`.
class TargetLoweringRegistry {
public:
  static TargetLoweringRegistry &get();

  /// Registers a lowering, taking ownership.
  void add(std::unique_ptr<TargetLowering> lowering);
  /// Returns the lowering matching `triple`, or null.
  const TargetLowering *lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetLowering>> targets() const {
    return Targets;
  }

private:
  TargetLoweringRegistry() = default;
  friend struct llvm::object_creator<TargetLoweringRegistry>;

  std::vector<std::unique_ptr<TargetLowering>> Targets;
};

/// Registers `LoweringT` at static-init, e.g.:
///   static RegisterTargetLowering<MyTargetLowering> X;
template <typename LoweringT>
struct RegisterTargetLowering {
  RegisterTargetLowering() {
    TargetLoweringRegistry::get().add(std::make_unique<LoweringT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H
