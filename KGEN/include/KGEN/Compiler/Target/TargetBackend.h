//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Per-target policy for LLVM-level codegen in ObjectCompiler: module splitting,
// runtime-library linking, object/asm emission, and packaging. Host and each
// GPU vendor are peer `TargetBackend`s, dispatched by triple via
// `TargetBackendRegistry`.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_TARGETBACKEND_H
#define KGEN_COMPILER_TARGET_TARGETBACKEND_H

#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace llvm {
class Function;
class Module;
class TargetMachine;
class Triple;
template <class C>
struct object_creator;
} // namespace llvm

namespace M::KGEN {

class MCLinker;
class PluginManager;

/// How the module is divided into independently-codegen'd units.
enum class SplitStrategy { None, PerExported, PerFunction };

/// Per-compilation state passed to the emit/package hooks.
struct EmitContext {
  /// Runs llc into `out`, emitting object code when `objectFile`, else asm.
  using RunLlc = llvm::function_ref<ErrorOrSuccess(
      llvm::Module &module, WriteableBuffer &out, bool objectFile)>;

  const CompilationOptions &options;
  llvm::TargetMachine &tm;
  mlir::Location loc;
  size_t moduleIdx = 0;
  MCLinker *linker = nullptr;
  PluginManager *pluginMgr = nullptr;
  RunLlc runLlc;
};

/// Immutable description of an LLVM-level compilation target.
class TargetBackend {
public:
  virtual ~TargetBackend() = default;

  /// Short name of the backend, e.g. "host", "nvptx", "amdgpu".
  virtual llvm::StringRef name() const = 0;
  /// Whether this backend handles `triple`.
  virtual bool matches(const llvm::Triple &triple) const = 0;

  virtual SplitStrategy
  splitStrategy(const CompilationOptions &options) const = 0;
  /// Inter-procedural codegen disables parallel/per-function llc.
  virtual bool isCodegenInterprocedural() const { return false; }

  // (TODO) revisit if this is needed
  virtual bool isOffload() const { return false; }

  /// Options used for `createTargetMachine`; the identity by default.
  virtual CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const {
    return options;
  }
  /// Fixes up `module` after the TargetMachine is created.
  virtual void finalizeModuleForTarget(llvm::Module &module,
                                       llvm::TargetMachine &tm,
                                       llvm::StringRef originalTriple) const {}

  /// Links target-specific device/runtime bitcode into `module`.
  virtual ErrorOrSuccess
  linkRuntimeLibraries(mlir::Location loc, llvm::Module &module,
                       const CompilationOptions &options) const {
    return {};
  }

  /// Attaches target-specific attributes to a kernel entry point.
  virtual void attachCodegenAttributes(llvm::Function *kernelEntry) const {}

  virtual ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                          EmitContext &ctx) const = 0;
  virtual ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                        EmitContext &ctx) const = 0;
  /// Combines per-unit objects into the final artifact.
  virtual ErrorOr<BufferRef>
  createArchive(llvm::MutableArrayRef<BufferRef> objects,
                llvm::StringRef moduleName, EmitContext &ctx) const = 0;

  virtual llvm::StringRef getAsmExtension() const = 0;
  virtual llvm::StringRef getLLVMExtension() const = 0;
  virtual llvm::StringRef getObjectExtension() const = 0;
};

/// Registry of `TargetBackend`s, dispatched by triple.
class TargetBackendRegistry {
public:
  static TargetBackendRegistry &get();

  /// Registers a backend, taking ownership.
  void add(std::unique_ptr<TargetBackend> backend);
  /// Returns the backend matching `triple`, or null.
  const TargetBackend *lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetBackend>> backends() const {
    return Backends;
  }

private:
  TargetBackendRegistry() = default;
  friend struct llvm::object_creator<TargetBackendRegistry>;

  std::vector<std::unique_ptr<TargetBackend>> Backends;
};

/// Registers `BackendT` at static-init, e.g.:
///   static RegisterTargetBackend<XYZBackend> X;
template <typename BackendT>
struct RegisterTargetBackend {
  RegisterTargetBackend() {
    TargetBackendRegistry::get().add(std::make_unique<BackendT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_TARGETBACKEND_H
