//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_OBJECTCOMPILER_H
#define KGEN_COMPILER_OBJECTCOMPILER_H

#include "Cache/BlobCache.h"
#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/PassManagerConfigOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallSet.h"
#include <filesystem>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
class DataLayout;
namespace orc {
class ExecutionSession;
} // namespace orc
} // namespace llvm

namespace M::KGEN {
struct SymbolAndMCInfo;

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects.
class ObjectCompiler {
public:
  /// Construct an ObjectCompiler that infers the exports from the module.
  static ErrorOr<std::unique_ptr<ObjectCompiler>>
  create(StringRef basePath, CompilationOptions options, bool isJIT,
         MLIRContext &context,
         PassManagerConfigOptions pmOptions = PassManagerConfigOptions());

  /// Emit the module to a object archive. If outKeyHash is provided, it will
  /// be populated with the hash of the key used to cache the module.
  ErrorOr<BufferRef> emitArchive(OwningOpRef<ModuleOp> module,
                                 bool emitAssembly = false,
                                 std::string *outKeyHash = nullptr);

  /// Lower the given module to LLVM. Returns the LLVM module on success, and
  /// nullptr on failure.
  ErrorOr<std::unique_ptr<llvm::Module>>
  lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// LLVMIR file. The LLVMIR output is written to the provided stream.
  ErrorOrSuccess emitLLVMIR(ModuleOp module, llvm::raw_pwrite_stream &os);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// assembly file. The assembly output is written to the provided stream.
  ErrorOrSuccess emitAssembly(OwningOpRef<ModuleOp> module,
                              llvm::raw_pwrite_stream &os);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// shared object file. The output is written to the provided stream.
  ErrorOrSuccess emitSharedObject(OwningOpRef<ModuleOp> module,
                                  llvm::raw_pwrite_stream &os);

  /// Writes C++ function declarations for all exported symbols.
  LogicalResult emitCXXHeader(ModuleOp module, StringRef filename,
                              raw_ostream &os);

  ErrorOr<DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>> emitGPUKernels(
      OwningOpRef<ModuleOp> module,
      llvm::DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> kernelEmissionKinds);

  /// Get a reference to the object compiler's transform cache.
  RCRef<Cache::TransformCache> getTransformCache() {
    return transformCache.copy();
  }

  /// Get whether compilation is for JIT.
  bool getIsJIT() const { return isJIT; }

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(
      RCRef<Cache::BlobCacheBackend> transformCache, CompilationOptions options,
      bool isJIT, MLIRContext &context, const std::string &linker,
      PassManagerConfigOptions pmOptions = PassManagerConfigOptions());

  /// Lower the given LLVM module to an object file (parLLC = false) or
  /// multiple object files per function (parLLC = true).
  AsyncRT::AsyncValueRef<SymbolAndMCInfo> lowerLLVMModuleToObjects(
      llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
      llvm::TargetMachine &targetMachine, bool parLLC,
      std::optional<size_t> moduleIdx, unsigned numFunctionsBase);

  /// Split llvm module and compile them in parallel towards the end of codegen
  /// but stop before AsmPrint. Return the MC compilation results.
  SmallVector<AsyncRT::AnyAsyncValueRef> emitArchiveParallelCompilation(
      LLVMModuleAndContext llvmModule, Location opLoc,
      llvm::TargetMachine &targetMachine,
      llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes);

  /// Link parallel compilation results and call AsmPrint to generate one object
  /// file.
  ErrorOr<WriteableBufferRef> emitArchiveMCLinking(
      MutableArrayRef<AnyAsyncValueRef> values, StringRef moduleName,
      bool emitAssembly,
      llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
      const llvm::StringMap<unsigned> &originalFnOrdering);

  /// Generate saveTempsPrefix files.
  ErrorOrSuccess emitArchiveSaveTemps(ModuleOp module, StringRef moduleName);

  /// The caches needed for compilation.
  RCRef<Cache::TransformCache> transformCache;

  /// The compilation options to use.
  CompilationOptions options;

  /// This is a bit odd, but since we use this layer to generate code for cases
  /// where we aren't going to immediately execute it, we need to be able to
  /// change the codegen mode.
  bool isJIT;

  /// PassManager configuration options.
  PassManagerConfigOptions pmOptions;

  /// The MLIR context.
  MLIRContext &context;

  /// The AsyncRT runtime.
  AsyncRT::Runtime &runtime;

  /// Mutex to protect deduplicating shared
  /// data structure among parallel splits.
  std::mutex dedupMutex;

  /// StringSet to deduplicate functions among parallel splits.
  llvm::StringSet<> seenCodeGenFns;

  /// Mutex to protect deduplicating TargetMachine to save peak memory
  /// footprint.
  std::mutex tmMutex;

  /// Name of the system linker.
  std::string linker;
};

/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(const CompilationOptions &options, bool isJIT);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_H
