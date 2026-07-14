//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Dependency-light per-target metadata, dispatched by triple via
// `TargetTraitsRegistry`. It lives in the low `KGEN/lib/Target` layer so any
// KGEN component can query a target's metadata without linking the codegen
// layer (`TargetBackend`). A supported target registers a full implementation
// from its own source file; a triple with no registered traits is an error at
// the use site (targets are dropped from a build by omitting their source).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_TARGETTRAITS_H
#define KGEN_TARGET_TARGETTRAITS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class Triple;
template <class C>
struct object_creator;
} // namespace llvm

namespace M::KGEN {

/// Per-target metadata dispatched by triple. Carries only cheap, broadly-useful
/// facts about a target (output-file extensions, accelerator arch tables); the
/// heavy codegen behavior stays on `TargetBackend` and the MLIR-lowering hooks
/// on `TargetLowering`.
class TargetTraits {
public:
  virtual ~TargetTraits() = default;

  /// Short name of the target, e.g. "host".
  virtual llvm::StringRef name() const = 0;
  /// Whether these traits describe `triple`.
  virtual bool matches(const llvm::Triple &triple) const = 0;

  /// Resolves `triple` to the concrete traits that describe it, or null if
  /// these do not. The default returns `this` when `matches`; a dispatcher that
  /// owns nested traits (e.g. one per loaded plugin) overrides this to return
  /// one of them. The registry prefers a resolution that differs from the
  /// traits it queried, so a nested traits object takes precedence over a
  /// direct match.
  virtual const TargetTraits *resolve(const llvm::Triple &triple) const {
    return matches(triple) ? this : nullptr;
  }

  /// Whether this target is a GPU.
  virtual bool isGPU() const { return false; }

  /// Default target CPU for `triple` when none is given, or empty to use the
  /// generic host/cross-compile fallback.
  virtual llvm::StringRef defaultCPU(const llvm::Triple &triple) const {
    return {};
  }

  /// Whether this target emits a standalone object file for offload kernels.
  /// Targets returning false have their emitted asm/IR file aliased under the
  /// object key instead (e.g. when the object toolchain is host-only).
  virtual bool emitsOffloadObjectFile() const { return true; }

  /// Required address space for a stack allocation on this target, or nullopt
  /// if the target does not constrain it.
  virtual std::optional<unsigned> requiredStackAllocationAddressSpace() const {
    return std::nullopt;
  }

  /// LLVM triple to use for this target's codegen, given `triple`. Defaults to
  /// `triple` unchanged; a target that compiles through a different LLVM triple
  /// normalizes it here.
  virtual std::string codegenTriple(llvm::StringRef triple) const {
    return triple.str();
  }

  /// Bitcode format version this target requires (an LLVM major version), or 0
  /// to use the caller's default. Overrides the caller-selected version.
  virtual unsigned forcedBitcodeVersion() const { return 0; }

  /// File extension for this target's assembly output (e.g. ".s").
  virtual llvm::StringRef getAsmExtension() const = 0;
  /// File extension for this target's LLVM IR output. Each target uses a
  /// distinct spelling so offload kernels from different targets do not
  /// collide in one output directory.
  virtual llvm::StringRef getLLVMExtension() const = 0;
  /// File extension for this target's object output (e.g. ".o").
  virtual llvm::StringRef getObjectExtension() const = 0;

  /// One accelerator architecture accepted by `--target-accelerator`.
  struct AcceleratorArch {
    llvm::StringRef arch;
    llvm::StringRef description;
  };

  /// Title of this target's section in the `--print-supported-accelerators`
  /// table. Targets with no accelerator archs return an empty title and are
  /// omitted from the table.
  virtual llvm::StringRef acceleratorSectionTitle() const { return {}; }
  /// The accelerator archs this target accepts, in display order.
  virtual llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const {
    return {};
  }
};

/// Registry of `TargetTraits`, dispatched by triple. Mirrors
/// `TargetLoweringRegistry` and `TargetBackendRegistry`.
class TargetTraitsRegistry {
public:
  static TargetTraitsRegistry &get();

  /// Registers a traits object, taking ownership.
  void add(std::unique_ptr<TargetTraits> traits);
  /// Returns the traits matching `triple`, or null.
  const TargetTraits *lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetTraits>> targets() const {
    return Targets;
  }

private:
  TargetTraitsRegistry() = default;
  friend struct llvm::object_creator<TargetTraitsRegistry>;

  std::vector<std::unique_ptr<TargetTraits>> Targets;
};

/// Registers `TraitsT` at static-init, e.g.:
///   static RegisterTargetTraits<HostTraits> X;
template <typename TraitsT>
struct RegisterTargetTraits {
  RegisterTargetTraits() {
    TargetTraitsRegistry::get().add(std::make_unique<TraitsT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_TARGETTRAITS_H
