//===- LLCL/Runtime/Algorithms.h ------------------------------------------===//
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
inline static void addTask(Runtime &runtime,
                           llvm::unique_function<void()> work) {
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
LLVM_NODISCARD inline static AsyncValueRef<ResultTy> addTask(Runtime &runtime,
                                                             FnTy work) {
  auto result = AsyncValueRef<ResultTy>::createUnconstructed(runtime);
  addTask(runtime,
          [result = result.copy(), work = std::forward<FnTy>(work)]() mutable {
            result.emplace(work());
          });
  return result;
}

//===----------------------------------------------------------------------===//
// parallelForEachN
//===----------------------------------------------------------------------===//

/// This version of parallelForEachN takes an element closure and a completion
/// handler callback.
template <typename... CaptureTys, typename ElementFn, typename CompletionFn>
static inline void
parallelForEachN(Runtime &runtime, size_t totalCount, ElementFn &&elementFn,
                 CompletionFn &&completionFn, CaptureTys &&...captures) {
  // If there is nothing to do, then we're already done.
  if (totalCount == 0)
    return;

  struct ParallelState {
    /// This is the number of elements left to finish executing.  When this
    /// drops to zero, the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute on each element.
    ElementFn elementFn;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// This is the state captured by the computation, it is passed to both the
    /// per-element computation as well as to the completion function.
    std::tuple<CaptureTys...> capturesList;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new ParallelState{totalCount,
                                 std::forward<ElementFn>(elementFn),
                                 std::forward<CompletionFn>(completionFn),
                                 {std::forward<CaptureTys>(captures)...}};

  // Enqueue each element of work!
  for (size_t elementIdx = 0; elementIdx != totalCount; ++elementIdx) {
    addTask(runtime, [state, elementIdx]() {
      // Invoke the per-element function with the index and all of the captured
      // state.
      std::apply([&](auto &&...args) { state->elementFn(elementIdx, args...); },
                 state->capturesList);
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      // Invoke the completion function, since we're done.
      llvm::apply_tuple([&](auto &&...args) { state->completionFn(args...); },
                        state->capturesList);

      // All uses of the state are done, so we can deallocate it.
      delete state;
    });
  }
}

} // namespace LLCL

#endif // LLCL_RUNTIME_ALGORITHMS_H
