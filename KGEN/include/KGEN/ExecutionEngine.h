//===- KGEN/ExecutionEngine.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_EXECUTION_ENGINE_H
#define KGEN_EXECUTION_ENGINE_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include <functional>

namespace M::KGEN {
namespace detail {
class ObjectCache;
} // namespace detail

/// This class provides an interface to interact with a compiled func. You
/// can either invoke the func, or get it as an object. The lifetime of one of
/// these objects is tied to the ExecutionEngine through the `cache` member.
/// This could be relaxed by using a pointer instead, but that would require
/// getObject to fail if the cache is unavailable, and there's currently no use
/// case for such a feature so we will leave it to the future.
class CompiledKernel {
public:
  /// Invoke this kernel. This has exactly the signature the compiled kernel
  /// does. Intended to have perfect forwarding of arguments into the
  /// function, and of return values from the function.
  template <typename ReturnT, typename... Args>
  ReturnT invoke(Args... args) {
    // Cast the function pointer and invoke it directly.
    return ((ReturnT(*)(Args...))fn)(std::forward<Args>(args)...);
  }

  /// Get the compiled object that corresponds to this kernel from `cache`.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> getObject();

private:
  /// Construct a CompiledKernel object. This constructor is private because it
  /// needs a reference to the cache that the ExecutionEngine holds, so it
  /// should really only be constructed from the ExecutionEngine or something
  /// like it.
  CompiledKernel(void *ptr, FuncOp func, detail::ObjectCache &cache)
      : fn(ptr), func(func), cache(cache) {}
  friend class ExecutionEngine;

  /// Pointer to the function to invoke.
  void *fn;

  /// This handle corresponds to this FuncOp.
  FuncOp func;

  /// Pointer to the object cache. This allows us to look up the compiled
  /// object as a memory buffer.
  detail::ObjectCache &cache;
};

/// This class provides an interface to the LLVM ORCJIT. It can compile
/// individual kernels (already lowered to the LLVM dialect) to object code. It
/// caches the objects themselves so we can retrieve them later and write them
/// to a file. The fundamental unit this class deals with is a single llvm
/// function because that's the minimum granularity we would want to use for
/// caching and search.
class ExecutionEngine {
public:
  ~ExecutionEngine();
  /// This class is move-constructible.
  ExecutionEngine(ExecutionEngine &&other);

  static ErrorOr<ExecutionEngine> create();

  /// Add an MLIR module to the execution engine. This will perform slicing for
  /// every kernel and generate self-contained libraries. `only` is a list of
  /// kernels to process, if empty, adds all the kernels in the module.
  ErrorOrSuccess add(mlir::ModuleOp module, ArrayRef<FuncOp> only = {});

  /// Look up a kernel and return it as a CompiledKernel object if we can find
  /// it.
  ErrorOr<CompiledKernel> lookup(KGEN::FuncOp kernel);

  /// Look up the opaque wrapper for a kernel and return it as a CompiledKernel
  /// object.
  ErrorOr<CompiledKernel> lookupOpaqueWrapper(KGEN::FuncOp kernel);

private:
  explicit ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit);

  /// This class is not copy-constructible.
  ExecutionEngine(const ExecutionEngine &other) = delete;

  llvm::orc::ThreadSafeContext ctx;
  std::unique_ptr<detail::ObjectCache> cache;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::vector<llvm::orc::ThreadSafeModule> compiledModules;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTION_ENGINE_H
