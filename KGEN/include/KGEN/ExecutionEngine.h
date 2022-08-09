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
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include <functional>

namespace M::KGEN {
namespace detail {
class ObjectCache;
} // namespace detail

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

    // Get the function pointer out of the ExecutorAddr.
    auto *fnPtr = fnOr->template toPtr<ReturnT (*)(Args...)>();

    // Invoke the function. If `ReturnT` is `void` then this will default to
    // returning a value of type `ErrorOrSuccess` set to `success()`. Otherwise,
    // it will return the result of the function, an object of type `ReturnT`.
    return invokeWithDefaultResultType<DefaultSuccess>(fnPtr, args...);
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
  getObject(mlir::LLVM::LLVMFuncOp kernel);

private:
  explicit ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit);

  /// This class is not copy-constructible.
  ExecutionEngine(const ExecutionEngine &other) = delete;

  llvm::orc::ThreadSafeContext ctx;
  std::unique_ptr<detail::ObjectCache> cache;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::unique_ptr<llvm::orc::LLJIT> jit;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTION_ENGINE_H
