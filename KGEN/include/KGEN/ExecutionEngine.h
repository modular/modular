//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_EXECUTION_ENGINE_H
#define KGEN_EXECUTION_ENGINE_H

#include "Cache/Buffer.h"
#include "KGEN/CompilationOptions.h"
#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/DataLayout.h"

namespace M::KGEN {
class CompilationOptions;

/// This class provides an interface to interact with a compiled func. You
/// can either invoke the func, or get it as an object. The lifetime of one of
/// these objects is tied to the ExecutionEngine through the `cache` member.
/// This could be relaxed by using a pointer instead, but that would require
/// getObject to fail if the cache is unavailable, and there's currently no use
/// case for such a feature so we will leave it to the future.
class CompiledFunc {
public:
  /// Invoke this func. This has exactly the signature the compiled func
  /// does. Intended to have perfect forwarding of arguments into the
  /// function, and of return values from the function.
  template <typename ReturnT, typename... Args>
  ReturnT invoke(Args... args) {
    // Cast the function pointer and invoke it directly.
    return ((ReturnT(*)(Args...))fn)(std::forward<Args>(args)...);
  }

  /// Return the pointer to the compiled function.
  void *getFunctionPointer() const { return fn; }

private:
  /// Construct a CompiledFunc object. This constructor is private because it
  /// needs a reference to the cache that the ExecutionEngine holds, so it
  /// should really only be constructed from the ExecutionEngine or something
  /// like it.
  CompiledFunc(void *ptr) : fn(ptr) {}
  friend class ExecutionEngine;

  /// Pointer to the function to invoke.
  void *fn;
};

/// This class provides an interface to the LLVM ORCJIT. It can compile
/// individual funcs (already lowered to the LLVM dialect) to object code. It
/// caches the objects themselves so we can retrieve them later and write them
/// to a file. The fundamental unit this class deals with is a single llvm
/// function because that's the minimum granularity we would want to use for
/// caching and search.
class ExecutionEngine {
public:
  ~ExecutionEngine();
  /// This class is move-constructible.
  ExecutionEngine(ExecutionEngine &&other);

  static ErrorOr<ExecutionEngine> create(const CompilationOptions &options);

  /// Add an archive to the JIT.
  ErrorOrSuccess add(StringRef libName, Cache::BufferRef archive);

  /// Add a function pointer to the JIT to participate in symbol resolution.
  // TODO (8082): This should not be necessary - the JIT should resolve things
  //   in the current process.
  ErrorOrSuccess add(StringRef libName, StringRef functionName, void *fn);

  /// Look up a func and return it as a CompiledFunc object if we can find it.
  ErrorOr<CompiledFunc> lookup(StringRef symbol);

private:
  explicit ExecutionEngine(CompilationOptions options,
                           std::unique_ptr<llvm::orc::ExecutionSession> session,
                           const llvm::DataLayout &dl);

  /// This class is not copy-constructible.
  ExecutionEngine(const ExecutionEngine &other) = delete;

  /// Get or create a JITDylib of name `libName`.
  ErrorOr<llvm::orc::JITDylib *> getOrCreateDylib(StringRef libName);

  /// Mangle and intern a string name.
  llvm::orc::SymbolStringPtr mangleAndIntern(StringRef name);

  void addToSearchOrder(StringRef name, llvm::orc::JITDylib *dylib);

  /// The compilation options to use.
  CompilationOptions options;

  /// The ORC requires an ExecutionSession - this is how it coordinates
  /// execution across processes/machines.
  std::unique_ptr<llvm::orc::ExecutionSession> executionSession = nullptr;

  /// JITLink linker. This is what drives all the linking underneath our JIT.
  std::unique_ptr<llvm::orc::ObjectLinkingLayer> objectLayer = nullptr;

  /// Keep a set of known dylibs and a dylib search order - this will make it
  /// easy to (a) make sure we only have unique dylibs and (b) cache the search
  /// order so we don't recreate it on every lookup.
  llvm::StringSet<> knownDylibs;
  llvm::orc::JITDylibSearchOrder searchOrder;

  llvm::DataLayout dataLayout;

  /// List of buffers that contain archive files added to the JIT. This holds
  /// references to them so they aren't deallocated underneath our feet.
  SmallVector<Cache::BufferRef> archiveBuffers;
};

/// This function is used to ensure the components of the orc are properly
/// linked.
LLVM_ATTRIBUTE_USED inline uintptr_t llvm_orc_dummyinit() {
  return (uintptr_t)&llvm_orc_registerJITLoaderGDBAllocAction +
         (uintptr_t)&llvm_orc_registerJITLoaderGDBWrapper;
}
} // namespace M::KGEN

#endif // KGEN_EXECUTION_ENGINE_H
