//===----------------------------------------------------------------------===//
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
#include "LLCL/Support/Chain.h"
#include "llvm/ADT/ArrayRef.h"
#include <utility>

namespace M::LLCL {

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
struct UnwrapErrorOr<ErrorOr<T>> {
  using type = T;
};

template <typename F>
using ResultType = typename UnwrapErrorOr<std::invoke_result_t<F>>::type;
} // namespace Detail

//===----------------------------------------------------------------------===//
// Helpers that wait for values.
//===----------------------------------------------------------------------===//

/// Donate the current thread to running work until all of the specified values
/// are ready.
inline static void await(llvm::ArrayRef<AnyAsyncValueRef> values) {
  if (!values.empty())
    values[0]->getRuntime()->getWorkQueue()->await(values);
}

template <typename T>
inline static void await(const AsyncValueRef<T> &value) {
  // Convert from a guaranteed AsyncValueRef to a guaranteed RCRef without
  // bumping reference counts.
  AnyAsyncValueRef ref = takeRCRef(value.getPointer());
  await(ref);
  (void)ref.release();
}

//===----------------------------------------------------------------------===//
// 'andThen' for multiple values with heterogenous types.
//===----------------------------------------------------------------------===//

/// This version of andThen takes a tuple of values to wait on, and passes the
/// elements into the completion handler as individual values.  It can be used
/// like this:
///
/// void example(AsyncValueRef<int32_t> lhs, AsyncValueRef<int32_t> rhs) {
///   ...
///   andThen(std::make_tuple(std::move(lhs), std::move(rhs)),
///           [... any captures...](AsyncValueRef<int32_t> lhs,
///                                 AsyncValueRef<int32_t> rhs) {
///     ... stuff that uses lhs/rhs ...
///   });
/// }
///
template <typename CompletionFn, typename... ValueTys>
inline static void andThen(std::tuple<ValueTys...> values,
                           CompletionFn completionFn) {
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
  auto state = new AndThenState{sizeof...(ValueTys),
                                std::forward<CompletionFn>(completionFn),
                                std::forward<decltype(values)>(values)};

  // This function is invoked on every async value to wait for it to complete.
  auto processAsyncValue = [&](AsyncValue *value) -> int {
    value->andThen([state]() {
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      // Invoke the completion function, since we're done.
      std::apply(state->completionFn, std::move(state->values));

      // All uses of the state are done, so we can deallocate it.
      delete state;
    });

    // Return an int just so the using make_tuple creates a tuple with trivial
    // element types.
    return 0;
  };

  // This magical incantation invokes `processAsyncValue` on each element of the
  // tuple.
  std::apply(
      [&](auto &...elt) {
        (void)std::make_tuple(processAsyncValue(elt.getPointer())...);
      },
      state->values);
}

//===----------------------------------------------------------------------===//
// 'andThen' for an array of values
//===----------------------------------------------------------------------===//

template <typename ArrayRefType, typename CompletionFn, typename CopyOrMoveFn>
inline static void andThenArrayImpl(ArrayRefType values,
                                    CompletionFn completionFn,
                                    CopyOrMoveFn copyOrMoveFn) {
  // Avoid malloc overhead for trivial cases.
  if (values.empty()) {
    completionFn(values);
    return;
  }
  if (values.size() == 1) {
    values[0]->andThen([completionFn = std::move(completionFn)](
                           const AnyAsyncValueRef &value) mutable {
      AnyAsyncValueRef mutableValue = value.copy();
      completionFn(mutableValue);
    });
    return;
  }

  struct AndThenState {
    /// This is the number of values we're waiting on.  When this drops to zero,
    /// the completion handler is run.
    std::atomic<size_t> numElementsLeft;

    /// This is the function to execute once all elements are done.
    CompletionFn completionFn;

    /// These are the async values we're waiting on.  They are passed into the
    /// completion function once they all become ready.
    llvm::SmallVector<AnyAsyncValueRef> values;
  };

  // Allocate the parallel state on the heap since it will out-live the call to
  // this function.  We will deallocate it after invoking the completion
  // handler when the last element completes.
  auto state = new AndThenState{
      values.size(), std::forward<CompletionFn>(completionFn), {}};

  state->values.reserve(values.size());

  // For each value, wait for completion and then run the completion function on
  // the last one.
  for (auto &v : values) {
    state->values.push_back(copyOrMoveFn(v));
    state->values.back()->andThen([state]() {
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      // Invoke the completion function, since we're done.
      state->completionFn(state->values);

      // All uses of the state are done, so we can deallocate it.
      delete state;
    });
  }
}

/// This version of andThen takes an array of homogenous AsyncValue references
/// to wait on, and passes the elements into the completion handler as
/// an ArrayRef.  It can be used like this:
///
/// void example() {
///   AsyncValueRef<int32_t> elements[4] = { ... };
///   ...
///   andThenCopying(elements,
///     [... any captures...](MutableArrayRef<AsyncValueRef<int32_t>> elts) {
///     ... stuff that uses elts ...
///   });
/// }
///
/// This is the "copying" form because it doesn't move the AsyncValue's from the
/// input array.
///
template <typename CompletionFn>
inline static void andThenCopying(llvm::ArrayRef<AnyAsyncValueRef> values,
                                  CompletionFn completionFn) {
  andThenArrayImpl(values, std::move(completionFn),
                   [](const AnyAsyncValueRef &ref) -> AnyAsyncValueRef {
                     return ref.copy();
                   });
}

/// This version of andThen takes an array of homogenous AsyncValue references
/// to wait on, and passes the elements into the completion handler as
/// an ArrayRef.  It can be used like this:
///
/// void example() {
///   AsyncValueRef<int32_t> elements[4] = { ... };
///   ...
///   andThenMoving(elements,
///     [... any captures...](MutableArrayRef<AsyncValueRef<int32_t>> elts) {
///     ... stuff that uses elts ...
///   });
/// }
///
/// This is the "moving" form because it destructively takes the elements out of
/// the array passed in.
///
template <typename CompletionFn>
inline static void andThenMoving(llvm::MutableArrayRef<AnyAsyncValueRef> values,
                                 CompletionFn completionFn) {
  andThenArrayImpl(
      values, std::move(completionFn),
      [](AnyAsyncValueRef &ref) -> AnyAsyncValueRef { return std::move(ref); });
}

//===----------------------------------------------------------------------===//
// Helpers to add tasks to the runtime's work queue
//===----------------------------------------------------------------------===//

/// Add some non-blocking work to the WorkQueue managed by the specified
/// Runtime.
template <typename FnTy, typename ResultTy = Detail::ResultType<FnTy>,
          std::enable_if_t<(std::is_void<ResultTy>()), int> = 0>
inline static void addTask(Runtime &runtime, FnTy f) {
  runtime.getWorkQueue()->addTask(std::forward<FnTy>(f));
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
[[nodiscard]] inline static AsyncValueRef<ResultTy> addTask(Runtime &runtime,
                                                            FnTy work) {
  auto result = AsyncValueRef<ResultTy>::allocate(runtime);

  addTask(runtime,
          [result = result.copy(), work = std::forward<FnTy>(work)]() mutable {
            result.emplace(work());
          });
  return result;
}

//===----------------------------------------------------------------------===//
// parallelForEachN
//===----------------------------------------------------------------------===//

namespace Detail {
/// Struct containing various utilities used by the implementation of
/// parallelForEachN.
struct ParallelForEachNUtils {
  /// A utility to build a tuple type containing the decay'd capture arguments
  /// of the element function of a parallelForEachN. The only non-general aspect
  /// of this is that it skips the first argument, which is the index of the
  /// element.
  template <typename... Ts>
  struct ElementFnCapturesImplT;
  template <typename FnTraitsT, size_t... Ns>
  struct ElementFnCapturesImplT<FnTraitsT, std::index_sequence<Ns...>> {
    using type =
        std::tuple<std::decay_t<typename FnTraitsT::template arg_t<Ns + 1>>...>;
  };
  template <typename FnT>
  using ElementFnCapturesT = typename ElementFnCapturesImplT<
      llvm::function_traits<FnT>,
      std::make_index_sequence<llvm::function_traits<FnT>::num_args - 1>>::type;
};

} // namespace Detail

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
    Detail::ParallelForEachNUtils::ElementFnCapturesT<ElementFn> capturesList;
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
      std::apply(
          [&](auto &&...args) { (void)state->elementFn(elementIdx, args...); },
          state->capturesList);
      // Once that is done we can decrement the count and trigger completion
      // when the last element is done.
      if (--state->numElementsLeft != 0)
        return;

      // Invoke the completion function, since we're done.
      std::apply(state->completionFn, state->capturesList);

      // All uses of the state are done, so we can deallocate it.
      delete state;
    });
  }
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// When all of the elements have finished, the `readyChain` is completed,
/// unblocking any computation `andThen`d on it.
///
template <typename... CaptureTys, typename ElementFn>
static inline void
parallelForEachNCompleteChain(Runtime &runtime, size_t totalCount,
                              AsyncValueRef<Chain> readyChain,
                              ElementFn &&elementFn, CaptureTys &&...captures) {
  parallelForEachNCustomCompletion(
      runtime, totalCount, std::forward<ElementFn>(elementFn),
      [readyChain = std::move(readyChain)](auto &&...args) {
        // When all the elements are ready, complete the `readyChain`,
        // unblocking other work.
        readyChain.emplace();
      },
      std::forward<CaptureTys...>(captures)...);
}

/// This method invokes the specified element function "N" times with indexes
/// from [0 ..< N).  This function returns immediately after kicking off the
/// work: all of the elements are processed on the Runtime's WorkQueue.
///
/// This helper takes an initialized `EltTy` value, and when complete it
/// emplaces it into the resultAV value.
///
template <typename EltTy, typename... CaptureTys, typename ElementFn>
static inline void
parallelForEachNFinishing(Runtime &runtime, size_t totalCount,
                          EltTy &&initialResultValue,
                          AsyncValueRef<EltTy> resultAV, ElementFn &&elementFn,
                          CaptureTys &&...captures) {
  parallelForEachNCustomCompletion(
      runtime, totalCount, std::forward<ElementFn>(elementFn),
      [resultAV = std::move(resultAV)](EltTy &result, auto &&...args) {
        // When all the elements are ready, emplace the result value into the
        // result AV.
        resultAV.emplace(std::move(result));
      },
      std::forward<EltTy>(initialResultValue),
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
  auto result = AsyncValueRef<Chain>::allocate(runtime);
  parallelForEachNCompleteChain(runtime, totalCount, result.copy(),
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

  if (totalCount == 0)
    return;

  // Execute N-1 elements for elements on background threads.
  AsyncValueRef<Chain> chainResult;
  if (totalCount > 1) {
    chainResult = parallelForEachNChain(
        runtime, totalCount - 1, std::forward<ElementFn>(elementFn),
        std::forward<CaptureTys...>(captures)...);
  }

  // Execute the last element on this thread since we'll be blocking otherwise.
  // This thread just spent a bunch of time kicking off work for other threads,
  // so it may be the straggler and a bit behind the rest of the pack. That
  // said, there is a reasonable likelihood that the last element will be
  // smaller than the rest, so this thread can catch up with the others.
  elementFn(totalCount - 1, captures...);

  // Donate the client thread to executing work until all the elements have
  // completed.
  if (chainResult)
    await(chainResult);
}

} // namespace M::LLCL

#endif // LLCL_RUNTIME_ALGORITHMS_H
