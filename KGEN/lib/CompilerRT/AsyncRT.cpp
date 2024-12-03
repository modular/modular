//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/TimerHeap.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Runtime/MojoCallContext.h"
#include "Runtime/MojoValue.h"
#include "Runtime/Tensor/StateContext.h"
#include "Runtime/Tensor/Tensor.h"
#include "Runtime/Tensor/TensorBufferRef.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorSpec.h"
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
using AsyncRTMojoCallContextRef = AsyncRTWrapper<MojoCallContext>;
using AsyncRTAsyncChainRef = AsyncRTWrapper<AsyncValueRef<Chain>>;
using AsyncRTSpinWaiterRef = AsyncRTWrapper<SpinWaiter<true>>;

/// Dummy entry point to force loading.
/// (All the other entry points use AsyncRTWrapper which we don't want to
/// have to include in the header).
COMPILERRT_EXPORT void KGEN_CompilerRT_AsyncRT_Dummy() {}

enum BorroweeType : size_t {
  kHandle = 0,
  kBuffer = 1,
  kTensor = 2,
};

//===----------------------------------------------------------------------===//
// Chains
//===----------------------------------------------------------------------===//

/// Creates a new AsyncValueRef<Chain> and assigns it to chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_InitializeChain(AsyncRTRuntimeRef rt,
                                        AsyncRTAsyncChainRef chain) {
  checkUniqueRuntime(unwrap(rt));
  new (&unwrap(chain))
      AsyncValueRef<Chain>(takeRCRef(AsyncValue::allocate<Chain>(unwrap(rt))));
}

/// Destroys the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_DestroyChain(AsyncRTAsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
}

/// Emplaces the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_Complete(AsyncRTAsyncChainRef chain) {
#if MODULAR_PARANOID
  unwrap(chain).getRuntime()->getWorkQueue()->taskIsDone();
#endif
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
                                AsyncRTRuntimeRef rt, ssize_t desiredWorkerId) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask(
      [resume, hdl] {
        resume(hdl);
#if MODULAR_PARANOID
        // Sleeping here gives any await loop the chance to exit and
        // proceed while this task is still 'active'. This can trigger
        // bugs since the common case is for the task to have returned
        // all the way up to the AsyncRT run items loop before any emplace
        // in the task body has been acted on.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
      },
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
                                       AsyncRTRuntimeRef rt,
                                       AsyncRTAsyncChainRef chain) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(unwrap(chain));
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_ExecuteAndResume(void (*resume)(int8_t *),
                                         int8_t *execHdl,
                                         AsyncRTAsyncChainRef chain,
                                         AsyncRTRuntimeRef rt,
                                         int8_t *resumeHdl) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask([resume, execHdl]() { resume(execHdl); });
  unwrap(chain).andThenAsync([resumeHdl, resume]() { resume(resumeHdl); });
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

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

/// Given a pointer to an AsyncRT runtime, destroy it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_DestroyRuntime(AsyncRTRuntimeRef rt) {
  delete &unwrap(rt);
}

/// Given a pointer to an AsyncRT runtime, get the number of threads in it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint32_t
KGEN_CompilerRT_AsyncRT_ParallelismLevel(AsyncRTRuntimeRef rt) {
  return unwrap(rt).getWorkQueue()->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// MojoCallContext
//===----------------------------------------------------------------------===//

/// Get cuda device from cuda runtime.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AsyncRT_MojoCallContext_GetDeviceContext(
    AsyncRTMojoCallContextRef callContext) {
  auto mojoValueDataPtr =
      reinterpret_cast<M::MojoValue *>(unwrap(callContext).deviceContext)
          ->getData();
  return mojoValueDataPtr;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_AsyncRT_MojoCallContext_Allocate(AsyncRTMojoCallContextRef ctx,
                                                 int64_t size,
                                                 int64_t alignment) {
  return unwrap(ctx).runtime->getAllocator()->allocateBytes(size, alignment);
}

//===----------------------------------------------------------------------===//
// Packing functions for creating async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error(
    AsyncRTMojoCallContextRef ctx, AsyncRTWrapper<AnyAsyncValueRef> *asyncs,
    size_t arrayLen, AsyncRTWrapper<Runtime> runtimePtr, const char *messagePtr,
    size_t messageLen) {
  StringRef errorMsg(messagePtr, messageLen);
  Runtime &runtime = unwrap(runtimePtr);
  // Set all async value ref to error;
  ArrayRef asyncArray(asyncs, arrayLen);
  for (AsyncRTWrapper<AnyAsyncValueRef> async : asyncArray) {
    AnyAsyncValueRef &value = unwrap(async);
    EncodedDiagnostic diagnostic{Twine(errorMsg), unwrap(ctx).loc.copy()};
    if (value.getPointer() && value.getPointer()->isIndirect())
      value.copy().setToError(std::move(diagnostic));
    else
      value = value.createError(runtime, std::move(diagnostic));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsync_bool(bool data,
                                 AsyncRTWrapper<AnyAsyncValueRef> async,
                                 AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<bool>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<bool>(runtime, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsync_ssizet(ssize_t data,
                                   AsyncRTWrapper<AnyAsyncValueRef> async,
                                   AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<size_t>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<size_t>(runtime, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsync_int64t(int64_t data,
                                   AsyncRTWrapper<AnyAsyncValueRef> async,
                                   AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<int64_t>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<int64_t>(runtime, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncVoidStar(void *data,
                                    AsyncRTWrapper<AnyAsyncValueRef> async,
                                    AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<void *>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<void *>(runtime, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsync_chain(AsyncRTWrapper<AnyAsyncValueRef> async,
                                  AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<Chain>();
  } else {
    assert(!value.isReady());
    value = value.createReady<Chain>(runtime);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncNonTrackedBufferRef(
    void *data, size_t size, AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorBufferRef>(
        TensorBufferRef::createWithNonTrackedMemory(
            data, size, /*alignment=*/std::nullopt));
  } else {
    assert(!value.isReady());
    value = value.createReady<TensorBufferRef>(
        runtime, TensorBufferRef::createWithNonTrackedMemory(
                     data, size, /*alignment=*/std::nullopt));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncBufferRef(void *data, size_t size,
                                     AsyncRTWrapper<AnyAsyncValueRef> async,
                                     AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorBufferRef>(
        ::M::TensorBufferRef::take(runtime, size, data));
  } else {
    assert(!value.isReady());
    value = value.createReady<TensorBufferRef>(
        runtime, ::M::TensorBufferRef::take(runtime, size, data));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncBufferWithBorrow(
    void *data, size_t size, AsyncRTWrapper<AnyAsyncValueRef> toBorrowFrom,
    size_t borroweeType, AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &outVal = unwrap(async);
  AnyAsyncValueRef &handleOrTensor = unwrap(toBorrowFrom);
  AnyAsyncValueRef handle;
  if (borroweeType == kHandle) {
    handle = std::move(handleOrTensor);
  } else if (borroweeType == kBuffer) {
    // Use the lifetime of the other tensor by sharing the same storage handle.
    handle = handleOrTensor.get<TensorBufferRef>().getMemStorageHandle();
  } else {
    assert(borroweeType == kTensor);
    handle = handleOrTensor.get<Tensor>().getBufferRef().getMemStorageHandle();
  }
  TensorBufferRef buf = ::M::TensorBufferRef::create(
      data, size, std::move(handle), std::optional<size_t>{});

  // Emplace into the async value.
  if (outVal.getPointer() && outVal.getPointer()->isIndirect()) {
    outVal.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    outVal = outVal.createReady<TensorBufferRef>(runtime, std::move(buf));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncTensorWithBorrow(
    void *data, size_t byteCount, size_t rank, size_t *dims, int8_t type,
    AsyncRTWrapper<AnyAsyncValueRef> toBorrowFrom, size_t borroweeType,
    AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &rt = unwrap(runtimePtr);
  AnyAsyncValueRef &outVal = unwrap(async);

  // Create a borrowed buffer ref.
  AnyAsyncValueRef &handleOrTensor = unwrap(toBorrowFrom);
  AnyAsyncValueRef handle;
  if (borroweeType == kHandle) {
    handle = std::move(handleOrTensor);
  } else if (borroweeType == kBuffer) {
    // Use the lifetime of the other tensor by sharing the same storage handle.
    handle = handleOrTensor.get<TensorBufferRef>().getMemStorageHandle();
  } else {
    assert(borroweeType == kTensor);
    handle = handleOrTensor.get<Tensor>().getBufferRef().getMemStorageHandle();
  }
  TensorBufferRef buf = ::M::TensorBufferRef::create(
      data, byteCount, std::move(handle), std::optional<size_t>{});

  // Pack buffer and spec into tensor.
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));
  if (outVal.getPointer() && outVal.getPointer()->isIndirect())
    outVal.copy().emplaceIndirect<Tensor>(std::move(buf), std::move(spec));
  else
    outVal = outVal.createReady<Tensor>(rt, std::move(buf), std::move(spec));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncMojoValueBufferRef(
    void *data, size_t size, void *mojoValue, void (*destructorFn)(void *),
    AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  AnyAsyncValueRef storageRef;
  storageRef =
      storageRef.createReady<MojoValue>(runtime, mojoValue, destructorFn);

  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorBufferRef>(TensorBufferRef::create(
        data, size, std::move(storageRef), /*alignment=*/1));
  } else {
    assert(!value.isReady());
    value = value.createReady<TensorBufferRef>(
        runtime, TensorBufferRef::create(data, size, std::move(storageRef),
                                         /*alignment=*/1));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsyncTensorSpec(ssize_t *data, ssize_t rank,
                                      int8_t rawDType,
                                      AsyncRTWrapper<AnyAsyncValueRef> async,
                                      AsyncRTWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  llvm::SmallVector<ssize_t> dims;
  for (int i = 0; i < rank; ++i)
    dims.push_back(data[i]);

  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorSpec>(TensorSpec(dims, DType(rawDType)));
  } else {
    assert(!value.isReady() &&
           "Value needs to not be ready so we can construct it.");
    value = AnyAsyncValueRef::createReady<TensorSpec>(
        runtime, TensorSpec(dims, DType(rawDType)));
  }
}

static void
createAsyncMojoValue(void *data, void (*destructorFn)(void *),
                     AsyncRTWrapper<AnyAsyncValueRef> async,
                     AsyncRTWrapper<Runtime> runtimePtr,
                     MojoValue::Tag tag = MojoValue::Tag::kDefault) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<MojoValue>(data, destructorFn, tag);
  } else {
    assert(!value.isReady());
    value = AnyAsyncValueRef::createReady<MojoValue>(runtime, data,
                                                     destructorFn, tag);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateOwnedAsyncMojoValue(
    void *data, void (*destructorFn)(void *),
    AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  createAsyncMojoValue(data, destructorFn, async, runtimePtr);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateOwnedAsyncPythonMojoValue(
    void *data, void (*destructorFn)(void *),
    AsyncRTWrapper<AnyAsyncValueRef> async,
    AsyncRTWrapper<Runtime> runtimePtr) {
  createAsyncMojoValue(data, destructorFn, async, runtimePtr,
                       MojoValue::Tag::kPython);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_MojoValueAllocateBuffer(size_t size, size_t align) {
  return MojoValue::allocateBuffer(size, align);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_MojoValueFreeBuffer(void *ptr) {
  MojoValue::freeBuffer(ptr);
}

//===----------------------------------------------------------------------===//
// Unpacking functions for reading async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetValueFromAsync(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  if (value.getPointer() && value.getPointer()->isIndirect())
    return value.getPointer()->getUnderlyingPtr();
  else
    return value.getPointerToData();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetDataFromBuffer(AsyncRTWrapper<AnyAsyncValueRef> async,
                                  size_t *sizeOut) {
  AnyAsyncValueRef &value = unwrap(async);
  auto &buffer = value.get<TensorBufferRef>();
  *sizeOut = buffer.getSize();
  return buffer.getBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint8_t
KGEN_CompilerRT_GetTensorSpecFromAsync(ssize_t *data, ssize_t rank,
                                       AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &spec = value.get<TensorSpec>();
  for (int i = 0; i < rank; ++i)
    data[i] = spec[i];
  return spec.getEltType().getValue();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetShapeAndDataFromTensor(
    size_t *shape, AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &tensor = value.get<Tensor>();
  const TensorSpec &spec = tensor.getSpec();
  for (size_t i = 0; i < spec.getRank(); ++i)
    shape[i] = spec[i];
  return tensor.getMutableBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetContextPayloadPtr(size_t index,
                                     AsyncRTWrapper<StateContext> rawCtx) {
  StateContext &ctx = unwrap(rawCtx);
  return ctx.getStateSlot(index).getUnderlyingPointer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetContextAndSizeFromAsync(
    size_t *size, AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &ctx = value.get<StateContext>();
  *size = ctx.getNumStateSlots();
  return reinterpret_cast<void *>(&ctx);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetCachedBuffer(size_t bBufferSlot,
                                AsyncRTWrapper<StateContext> rawCtx,
                                size_t *size, AnyAsyncValueRef *storageRefPtr) {
  static_assert(sizeof(AnyAsyncValueRef) == sizeof(void *) &&
                "SANITY CHECK FAILED: Graph Compiler allocates a `void *` to "
                "hold the async value ref, keep two sizes consistent to make "
                "the function work.");
  assert(storageRefPtr != nullptr);

  StateContext &theContext = unwrap(rawCtx);
  auto &stateSlot = theContext.getStateSlot(bBufferSlot);
  ErrorOr<TensorBufferRef> errOr = stateSlot.getBuffer(/*bufferIndex=*/0);
  if (errOr)
    return nullptr;

  TensorBufferRef ref = errOr.takeValue();
  // Emplace construction using the storage_ref of the TensorBufferRef.
  new ((void *)storageRefPtr) AnyAsyncValueRef(ref.getMemStorageHandle());
  *size = ref.getSize();
  return ref.getBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_RemoveCachedBuffer(size_t bufferSlot,
                                   AsyncRTWrapper<StateContext> rawCtx) {
  StateContext &theContext = unwrap(rawCtx);
  auto &stateSlot = theContext.getStateSlot(bufferSlot);
  stateSlot.clear();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_DestructAsyncRefs(size_t size, void **storageRefPtr) {
  for (size_t i = 0; i < size; i++) {
    void *ref_addr = storageRefPtr[i];
    std::destroy_at(reinterpret_cast<AnyAsyncValueRef *>(ref_addr));
  }
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
