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
// Helpers that wait for values.
//===----------------------------------------------------------------------===//

/// Donate the current thread to running work until all of the specified values
/// are ready.
inline static void await(Runtime &runtime,
                         llvm::ArrayRef<RCRef<AsyncValue>> values) {
  runtime.getWorkQueue()->await(values);
}

template <typename T>
inline static void await(Runtime &runtime, const AsyncValueRef<T> &value) {
  // Convert from a guaranteed AsyncValueRef to a guaranteed RCRef without
  // bumping reference counts.
  RCRef<AsyncValue> ref = takeRCRef(value.getPointer());
  runtime.getWorkQueue()->await(ref);
  (void)ref.release();
}

//===----------------------------------------------------------------------===//
// 'andThen' for multiple values.
//===----------------------------------------------------------------------===//

template <typename CompletionFn, typename... ValueTys>
inline static void andThen(CompletionFn completionFn, ValueTys &&...values) {
  struct AndThenState {
    /// This is the number of values we're waiting on.  When this drops to zero,
    /// the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// These are the async values we're waiting on.  They are passed into the
    /// completion function once they all become ready.
    std::tuple<ValueTys...> values;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new AndThenState{sizeof...(values),
                                std::forward<CompletionFn>(completionFn),
                                {std::forward<ValueTys>(values)...}};

  // This function is invoked on every async value to wait for it to complete.
  auto processAsyncValue = [&](AsyncValue *value) -> int {
    value->andThen([state]() {
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      // Invoke the completion function, since we're done.
      llvm::apply_tuple(
          [&](auto &&...args) {
            state->completionFn(std::forward<ValueTys>(args)...);
          },
          state->values);

      // All uses of the state are done, so we can deallocate it.
      delete state;
    });

    // Return an int just so the using make_tuple creates a tuple with trivial
    // element types.
    return 0;
  };

  // This magical incantation invokes `processAsyncValue` on each element of the
  // tuple.
  llvm::apply_tuple(
      [&](auto &...elt) {
        (void)std::make_tuple(processAsyncValue(elt.getPointer())...);
      },
      state->values);
}

//===----------------------------------------------------------------------===//
// Helpers to add tasks to the runtime's work queue
//===----------------------------------------------------------------------===//

/// Add some non-blocking work to the WorkQueue managed by the specified
/// Runtime.
inline static void addTask(Runtime &runtime,
                           llvm::unique_function<void()> work) {
  runtime.getWorkQueue()->addTask(std::move(work));
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

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, a completion handler is invoked.
///
template <typename... CaptureTys, typename ElementFn, typename CompletionFn>
static inline void parallelForEachNCustomCompletion(Runtime &runtime,
                                                    size_t totalCount,
                                                    ElementFn &&elementFn,
                                                    CompletionFn &&completionFn,
                                                    CaptureTys &&...captures) {
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

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, the `readyMarker` is marked as
/// ready, unblocking any computation `andThen`d on it.  Its `AsyncValue` may
/// contain any type.
///
template <typename... CaptureTys, typename ElementFn>
static inline void
parallelForEachNMarkReady(Runtime &runtime, size_t totalCount,
                          RCRef<AsyncValue> readyMarker, ElementFn &&elementFn,
                          CaptureTys &&...captures) {
  parallelForEachNCustomCompletion(
      runtime, totalCount, std::forward<ElementFn>(elementFn),
      [readyMarker = std::move(readyMarker)](auto &&...args) {
        // When all the elements are ready, mark the `readyMarker` as complete,
        // unblocking other work.
        readyMarker->markReady();
      },
      std::forward<CaptureTys...>(captures)...);
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, the chain result is marked as ready.
/// provides a convenient way to chain together work with `.andThen` on the
/// chain.
///
template <typename... CaptureTys, typename ElementFn>
static inline AsyncValueRef<Chain>
parallelForEachNChain(Runtime &runtime, size_t totalCount,
                      ElementFn &&elementFn, CaptureTys &&...captures) {
  auto result = AsyncValueRef<Chain>::createConstructed(runtime);
  parallelForEachNMarkReady(runtime, totalCount, result.copy(),
                            std::forward<ElementFn>(elementFn),
                            std::forward<CaptureTys...>(captures)...);
  return result;
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function kicks off the per-element work into the
/// Runtime's WorkQueue and then donates the client thread to doing work.  It
/// returns when all the elements are completed.
///
/// Because this doesn't return until the elements are done, it is ok for the
/// element function to capture things on the caller's stack by reference.
///
template <typename... CaptureTys, typename ElementFn>
static inline void parallelForEachN(Runtime &runtime, size_t totalCount,
                                    ElementFn &&elementFn,
                                    CaptureTys &&...captures) {
  auto chainResult = parallelForEachNChain(
      runtime, totalCount, std::forward<ElementFn>(elementFn),
      std::forward<CaptureTys...>(captures)...);

  // Donate the client thread to executing work until all the elements have
  // completed.
  await(runtime, chainResult);
}

} // namespace LLCL

#endif // LLCL_RUNTIME_ALGORITHMS_H
