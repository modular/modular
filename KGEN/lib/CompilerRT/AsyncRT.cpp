//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/TimerHeap.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Runtime/MojoValue.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

using namespace M;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An opaque wrapper around a pointer to a T.
template <typename T>
struct AsyncRTWrapper {
  void *ptr;
};

template <typename T>
T &unwrap(AsyncRTWrapper<T> ref) {
  return *reinterpret_cast<T *>(ref.ptr);
}
template <typename T>
AsyncRTWrapper<T> wrap(T *ptr) {
  return AsyncRTWrapper<T>{ptr};
}

using AsyncRTRuntimeRef = AsyncRTWrapper<Runtime>;
using AsyncRTAsyncChainRef = AsyncRTWrapper<AsyncValueRef<Chain>>;
using AsyncRTSpinWaiterRef = AsyncRTWrapper<SpinWaiter<true>>;

//===----------------------------------------------------------------------===//
// Chains
//===----------------------------------------------------------------------===//

/// Creates a new AsyncValueRef<Chain> and assigns it to chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_InitializeChain(AsyncRTAsyncChainRef chain) {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  new (&unwrap(chain))
      AsyncValueRef<Chain>(takeRCRef(AsyncValue::allocate<Chain>(rt)));
}

/// Destroys the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_DestroyChain(AsyncRTAsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
}

/// Emplaces the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_Complete(AsyncRTAsyncChainRef chain) {
  unwrap(chain).copy().emplace();
}

/// Blocks until the given chain is ready.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_Wait(AsyncRTAsyncChainRef chain) {
  await(unwrap(chain));
}

/// Blocks until the given chain is ready, or the given deadline is hit.
///
/// The timeout provided is in nanoseconds. True is returned if the value is
/// ready, false is a timeout occurred. Note that the value may be ready by the
/// time the function returns regardless.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT bool
KGEN_CompilerRT_AsyncRT_Wait_Timeout(AsyncRTAsyncChainRef chain, int64_t ns) {
  static TimerHeap heap;
  AsyncValueRef<Chain> &done = unwrap(chain);
  AsyncValueRef<Chain> expired =
      AsyncValueRef<Chain>::allocate(done.getRuntime());
  AsyncValueRef<Chain> either =
      AsyncValueRef<Chain>::allocate(done.getRuntime());

  // Compute the expiration and push it to the heap.
  TimerHeap::deadline expiration =
      std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
  heap.push(expiration, expired);

  // Wait for either, return true if our wrapped chain is ready. Unfortunately
  // we have to have a separate shared allocation and two additional anonymous
  // functions. This is quite inefficient, and can be improved in the future.
  auto emplaced = std::make_shared<std::atomic<bool>>();
  done.andThenSync([emplaced, either = either.copy()]() mutable {
    if (!emplaced->exchange(true))
      std::move(either).emplace();
  });
  expired.andThenSync([emplaced, either = either.copy()]() mutable {
    if (!emplaced->exchange(true))
      std::move(either).emplace();
  });

  await(either);
  if (done.isReady()) {
    heap.cancel(expired);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine as an AsyncRT task on the given runtime. If
/// desiredWorkerId is >= 0 then the task will be executed by the worker thread
/// with that id. Otherwise the task will be executed by the next available
/// worker thread. Scheduling tasks onto specific workers can avoid some AsyncRT
/// scheduling overhead and ensure worker's are balanced.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_Execute(void (*resume)(int8_t *), int8_t *hdl,
                                ssize_t desiredWorkerId) {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  rt->getWorkQueue()->addTask([resume, hdl] { resume(hdl); },
                              static_cast<int>(desiredWorkerId));
}

/// Resume a coroutine when the current one completes.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_AndThen(void (*resume)(int8_t *),
                                AsyncRTAsyncChainRef chain, int8_t *hdl) {
  unwrap(chain).andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Execute a coroutine and block the current routine until it is complete.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_ExecuteAndWait(void (*resume)(int8_t *), int8_t *hdl,
                                       AsyncRTAsyncChainRef chain) {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  checkUniqueRuntime(*rt);
  rt->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(unwrap(chain));
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_ExecuteAndResume(void (*resume)(int8_t *),
                                         int8_t *execHdl,
                                         AsyncRTAsyncChainRef chain,
                                         int8_t *resumeHdl) {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  checkUniqueRuntime(*rt);
  rt->getWorkQueue()->addTask([resume, execHdl]() { resume(execHdl); });
  unwrap(chain).andThenAsync([resumeHdl, resume]() { resume(resumeHdl); });
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// Given a pointer to an AsyncRT runtime, destroy it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_DestroyRuntime(AsyncRTRuntimeRef rt) {
  delete &unwrap(rt);
}

/// Returns the pointer to the runtime to which the caller's thread is
/// associated. Returns null if the caller's thread is not managed by any
/// runtime.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncRTRuntimeRef
KGEN_CompilerRT_AsyncRT_GetCurrentRuntime() {
  return wrap(Runtime::getCurrentRuntimeOrNull());
}

/// Create an AsyncRT runtime and return its pointer.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncRTRuntimeRef
KGEN_CompilerRT_AsyncRT_CreateRuntime(ssize_t numThreads) {
  // Create non global runtimes from mojo with mainWillDonate=false. Refer to
  // Runtime.h for detailed explanation.
  auto options = numThreads > 0 ? RuntimeOptions().withMainWillNotDonate()
                                : RuntimeOptions();
  auto runtime = createUniqueRuntime(options.withNumThreads(numThreads));
  return wrap(runtime.release());
}

/// Given a pointer to an AsyncRT runtime, get the number of threads in it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint32_t
KGEN_CompilerRT_AsyncRT_ParallelismLevel() {
  auto rt = Runtime::getCurrentRuntimeOrNull();
  return rt->getWorkQueue()->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// Packing functions for creating async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error(
    AsyncRTWrapper<AnyAsyncValueRef> *asyncs, size_t arrayLen,
    const char *messagePtr, size_t messageLen) {
  StringRef errorMsg(messagePtr, messageLen);
  Runtime &runtime = *Runtime::getCurrentRuntimeOrNull();
  // Set all async value ref to error;
  ArrayRef asyncArray(asyncs, arrayLen);
  for (AsyncRTWrapper<AnyAsyncValueRef> async : asyncArray) {
    AnyAsyncValueRef &value = unwrap(async);

    EncodedDiagnostic diagnostic{Twine(errorMsg),
                                 UnknownLocationDecoder::getEncodedLocation()};
    if (value.getPointer() && value.getPointer()->isIndirect())
      value.copy().setToError(std::move(diagnostic));
    else
      value = value.createError(runtime, std::move(diagnostic));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncVoidStar(void *data,
                                    AsyncRTWrapper<AnyAsyncValueRef> async) {
  Runtime &runtime = *Runtime::getCurrentRuntimeOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<void *>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<void *>(runtime, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_MojoValueFreeBuffer(void *ptr) {
  MojoValue::freeBuffer(ptr);
}

//===----------------------------------------------------------------------===//
// SpinWaiter function
//===----------------------------------------------------------------------===//

/// Creates a new SpinWaiter instance.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AsyncRT_InitializeSpinWaiter() {
  return new SpinWaiter<true>();
}

/// Waits on the SpinWaiter
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_SpinWaiter_Wait(AsyncRTSpinWaiterRef waiter) {
  unwrap(waiter).wait();
}

/// Destroys the given SpinWaiter.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_DestroySpinWaiter(AsyncRTSpinWaiterRef waiter) {
  delete (SpinWaiter<true> *)(waiter.ptr);
}
