//===- LLCL/Runtime/Algorithms.h - Parallel Algorithms ----------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares global functions that help implement parallel algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_ALGORITHMS_H
#define LLCL_RUNTIME_ALGORITHMS_H

#include "LLCL/Runtime/Runtime.h"

namespace LLCL {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace Detail {
// Extract the result type of a function passed to addTask(Runtime, fn).
template <typename T>
struct UnwrapErrorOr {
  using type = T;
};
template <typename T>
struct UnwrapErrorOr<M::ErrorOr<T>> {
  using type = T;
};

template <typename F>
using ResultType = typename UnwrapErrorOr<std::result_of_t<F()>>::type;
} // namespace Detail

//===----------------------------------------------------------------------===//
// Helpers to add tasks to the runtime's work queue
//===----------------------------------------------------------------------===//

/// Add some non-blocking work to the WorkQueue managed by the specified
/// Runtime.
static void addTask(Runtime &runtime, llvm::unique_function<void()> work) {
  runtime.addTask(std::move(work));
}

/// Overload of addTask that returns AsyncValueRef<R> for work that returns R
/// (when R is not void).
///
/// Example:
/// int a = 1, b = 2;
/// AsyncValueRef<int> r = addTask(runtime, [a, b] { return a + b; });
///
template <typename FnTy, typename ResultTy = Detail::ResultType<FnTy>,
          std::enable_if_t<!std::is_void<ResultTy>(), int> = 0>
LLVM_NODISCARD static AsyncValueRef<ResultTy> addTask(Runtime &runtime,
                                                      FnTy work) {
  auto result = AsyncValueRef<ResultTy>::createUnconstructed(runtime);
  addTask(runtime,
          [result = result.copy(), work = std::forward<FnTy>(work)]() mutable {
            result.emplace(work());
          });
  return result;
}

} // namespace LLCL

#endif // LLCL_RUNTIME_ALGORITHMS_H
