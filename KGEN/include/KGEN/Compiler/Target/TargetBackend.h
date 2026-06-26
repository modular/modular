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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

#include <memory>
#include <vector>

namespace llvm {
class Function;
class Module;
class PassBuilder;
class TargetMachine;
class Triple;
class raw_pwrite_stream;
template <class C>
struct object_creator;
} // namespace llvm

namespace mlir {
class Operation;
} // namespace mlir

namespace M::KGEN {

class MCLinker;
class PluginManager;

/// Creates an LLVM TargetMachine from already-effective `options` (no
/// target-specific adjustment). Used by `TargetBackend::createTargetMachine`
/// and as the fallback for triples with no registered backend.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
defaultCreateTargetMachine(const CompilationOptions &options, bool isJIT);

/// How the module is divided into independently-codegen'd units.
enum class SplitStrategy { None, PerExported, PerFunction };

/// Per-compilation state passed to the emit/package hooks.
struct EmitContext {
  /// Runs llc into `out`, emitting object code when `objectFile`, else asm.
  using RunLlc = llvm::function_ref<ErrorOrSuccess(
      llvm::Module &module, WriteableBuffer &out, bool createObjectFile)>;
  /// Links an emitted object into the final shared object/binary (plugin or
  /// lld).
  using LinkObject = llvm::function_ref<ErrorOr<BufferRef>(
      BufferRef object, llvm::StringRef moduleName)>;

  const CompilationOptions &options;
  llvm::TargetMachine &tm;
  mlir::Location loc;
  size_t moduleIdx = 0;
  MCLinker *linker = nullptr;
  PluginManager *pluginMgr = nullptr;
  RunLlc runLlc;
  LinkObject linkObject;
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

  /// Whether this backend produces offload (device) code; a missing kernel id
  /// is a hard error for offload targets.
  virtual bool isOffload() const { return false; }

  /// Whether the linked module's functions must be restored to their original
  /// order after MCLinker reorders them (needed when codegen emits function
  /// declarations whose order matters, e.g. NVPTX).
  virtual bool requiresOriginalFunctionOrder() const { return false; }

  /// Adjusts the MLIR module before lowering to LLVM (e.g. target-specific
  /// debug-info fixups). Runs in `ObjectCompiler::lowerAllFuncsToLLVM`.
  virtual void
  prepareModuleForLowering(mlir::Operation *module,
                           const CompilationOptions &options) const {}

  /// Options used for `createTargetMachine`; the identity by default.
  virtual CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const {
    return options;
  }
  /// Creates the LLVM TargetMachine for `options`. The base implementation
  /// applies `adjustOptionsForTargetMachine` and uses the generic
  /// TargetRegistry path; backends may override for full control.
  virtual ErrorOr<std::unique_ptr<llvm::TargetMachine>>
  createTargetMachine(const CompilationOptions &options, bool isJIT) const;
  /// Fixes up `module` after the TargetMachine is created.
  virtual void finalizeModuleForTarget(llvm::Module &module,
                                       llvm::TargetMachine &tm,
                                       llvm::StringRef originalTriple) const {}

  /// Writes `module` as bitcode to `os`. The base implementation writes
  /// standard LLVM bitcode; backends may override (e.g. Metal emits AIR
  /// bitcode).
  virtual void emitBitcode(llvm::Module &module,
                           llvm::raw_pwrite_stream &os) const;

  /// Builds the backend's complete LLVM pass pipeline into `mpm`, returning
  /// true if it fully owns the pipeline (so the generic optimization pipeline
  /// is skipped). The default returns false; backends that need a fully custom
  /// pipeline (e.g. Metal's AIR legalization) override it.
  virtual bool buildLLVMPipeline(llvm::ModulePassManager &mpm,
                                 llvm::PassBuilder &passBuilder,
                                 const CompilationOptions &options) const {
    return false;
  }

  /// Adds backend-specific passes at the start of the standard optimization
  /// pipeline (for backends that augment rather than replace it, e.g. NVPTX).
  virtual void addPipelineStartPasses(llvm::ModulePassManager &mpm,
                                      const CompilationOptions &options) const {
  }

  /// Registers the backend's named passes with `passBuilder` for `-passes=`
  /// (used by kgen-llvm-opt). The default registers nothing.
  virtual void registerPipelinePasses(llvm::PassBuilder &passBuilder) const {}

  /// Links target-specific device/runtime bitcode into `module`.
  virtual ErrorOrSuccess
  linkRuntimeLibraries(mlir::Location loc, llvm::Module &module,
                       const CompilationOptions &options) const {
    return {};
  }

  /// Attaches target-specific attributes to a kernel entry point.
  virtual void attachCodegenAttributes(llvm::Function *kernelEntry) const {}

  /// Appends backend-specific arguments to the link step.
  virtual void appendLinkArgs(llvm::SmallVectorImpl<llvm::StringRef> &args,
                              const CompilationOptions &options) const {}

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
