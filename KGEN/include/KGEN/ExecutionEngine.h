//===- KGEN/ExecutionEngine.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_EXECUTION_ENGINE_H
#define KGEN_EXECUTION_ENGINE_H

#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include <functional>

namespace M::KGEN {
namespace detail {
/// Provides a simple object cache. Users shouldn't be interacting directly with
/// this cache, they should interact with `ExecutionEngine` below.
class ObjectCache : public llvm::ObjectCache {
public:
  /// notifyObjectCompiled - Provides a pointer to compiled code for Module M.
  void notifyObjectCompiled(const llvm::Module *M,
                            llvm::MemoryBufferRef Obj) override;

  /// Returns a pointer to a newly allocated MemoryBuffer that contains the
  /// object which corresponds with Module M, or 0 if an object is not
  /// available.
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *M) override;

  std::unique_ptr<llvm::MemoryBuffer> getObject(llvm::StringRef name);

  /// Check if the cache has the object with the given name.
  bool hasObject(llvm::StringRef name) { return storage.count(name) != 0; }

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> storage;
};
} // namespace detail

/// This class provides an interface to the LLVM ORCJIT. It can compile
/// individual kernels (already lowered to the LLVM dialect) to object code. It
/// caches the objects themselves so we can retrieve them later and write them
/// to a file. The fundamental unit this class deals with is a single llvm
/// function because that's the minimum granularity we would want to use for
/// caching and search.
class ExecutionEngine {
public:
  static ErrorOr<ExecutionEngine> create();

  /// Add a function to the executor. This will compile the LLVM function
  /// immediately. This is safe to call multiple times on a single kernel - if
  /// we already have the kernel in the object cache then we won't recompile it.
  /// This produces individual objects that are somewhat self-contained in that
  /// they will not depend on any other object in the object cache.
  ErrorOrSuccess add(mlir::LLVM::LLVMFuncOp kernel);

  /// Invoke a kernel with the given name. This will return an error if the
  /// kernel hasn't been previously added.
  template <typename ReturnT, typename... Args>
  ErrorOr<std::conditional_t<std::is_void_v<ReturnT>, SuccessType, ReturnT>>
  invoke(llvm::StringRef kernel, Args... args) {
    auto fnOr = jit->lookup(kernel);
    if (!fnOr)
      return Error(llvm::toString(fnOr.takeError()));

    return invokeWithDefaultResultType<DefaultSuccess>(
        reinterpret_cast<ReturnT (*)(Args...)>(fnOr->template toPtr<void *>()),
        args...);
  }

  template <typename ReturnT, typename... Args>
  auto invoke(mlir::LLVM::LLVMFuncOp kernel, Args... args) {
    return invoke<ReturnT, Args...>(kernel.getName(),
                                    std::forward<Args>(args)...);
  }

  /// Get the compiled object that corresponds to this kernel.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getObject(llvm::StringRef kernel);

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getObject(mlir::LLVM::LLVMFuncOp kernel) {
    return getObject(kernel.getName());
  }

private:
  ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit);

  llvm::orc::ThreadSafeContext ctx;
  std::unique_ptr<detail::ObjectCache> cache;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::unique_ptr<llvm::orc::LLJIT> jit;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTION_ENGINE_H
