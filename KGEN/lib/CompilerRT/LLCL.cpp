//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/StateContext.h"
#include "ArraySupport/TensorBufferRef.h"
#include "CUDASupport/CUDARuntime.h"
#include "CUDASupport/Globals/Globals.h"
#include "KGEN/CompilerRT/Registration.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/TimerHeap.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Runtime/MojoCallContext.h"
#include "Runtime/MojoValue.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

using namespace M;
using namespace M::LLCL;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An opaque wrapper around a pointer to a T.
template <typename T>
struct LLCLWrapper {
  void *ptr;
};

template <typename T>
T &unwrap(LLCLWrapper<T> ref) {
  return *reinterpret_cast<T *>(ref.ptr);
}
template <typename T>
LLCLWrapper<T> wrap(T *ptr) {
  return LLCLWrapper<T>{ptr};
}

using LLCLRuntimeRef = LLCLWrapper<Runtime>;
using LLCLMojoCallContextRef = LLCLWrapper<MojoCallContext>;
using LLCLAsyncChainRef = LLCLWrapper<AsyncValueRef<Chain>>;
using LLCLSpinWaiterRef = LLCLWrapper<SpinWaiter<true>>;

/// Dummy entry point to force loading.
/// (All the other entry points use LLCLWrapper which we don't want to
/// have to include in the header).
COMPILERRT_EXPORT void KGEN_CompilerRT_LLCL_Dummy() {}

//===----------------------------------------------------------------------===//
// Chains
//===----------------------------------------------------------------------===//

/// Creates a new AsyncValueRef<Chain> and assigns it to chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_InitializeChain(LLCLRuntimeRef rt,
                                     LLCLAsyncChainRef chain) {
  checkUniqueRuntime(unwrap(rt));
  new (&unwrap(chain))
      AsyncValueRef<Chain>(takeRCRef(AsyncValue::allocate<Chain>(unwrap(rt))));
}

/// Destroys the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_DestroyChain(LLCLAsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
}

/// Emplaces the given chain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Complete(LLCLAsyncChainRef chain) {
#if MODULAR_PARANOID
  unwrap(chain).getRuntime()->getWorkQueue()->taskIsDone();
#endif
  unwrap(chain).copy().emplace();
}

/// Blocks until the given chain is ready.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Wait(LLCLAsyncChainRef chain) {
  await(unwrap(chain));
}

/// Blocks until the given chain is ready, or the given deadline is hit.
///
/// The timeout provided is in nanoseconds. True is returned if the value is
/// ready, false is a timeout occurred. Note that the value may be ready by the
/// time the function returns regardless.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT bool
KGEN_CompilerRT_LLCL_Wait_Timeout(LLCLAsyncChainRef chain, int64_t ns) {
  static TimerHeap heap;
  AsyncValueRef<Chain> &done = unwrap(chain);
  AsyncValueRef<Chain> expired =
      AsyncValueRef<Chain>::allocate(done.getRuntime());
  AsyncValueRef<Chain> either =
      AsyncValueRef<Chain>::allocate(done.getRuntime());

  // Compute the expiration and push it to the heap.
  TimerHeap::deadline expiration =
      std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
  heap.push(expiration, expired.copy());

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
  return done.isReady();
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine as an LLCL task on the given runtime. If desiredWorkerId
/// is >= 0 then the task will be executed by the worker thread with that id.
/// Otherwise the task will be executed by the next available worker thread.
/// Scheduling tasks onto specific workers can avoid some LLCL scheduling
/// overhead and ensure worker's are balanced.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *), int8_t *hdl,
                             LLCLRuntimeRef rt, ssize_t desiredWorkerId) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask(
      [resume, hdl] {
        resume(hdl);
#if MODULAR_PARANOID
        // Sleeping here gives any await loop the chance to exit and
        // proceed while this task is still 'active'. This can trigger
        // bugs since the common case is for the task to have returned
        // all the way up to the LLCL run items loop before any emplace
        // in the task body has been acted on.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
      },
      static_cast<int>(desiredWorkerId));
}

/// Resume a coroutine when the current one completes.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_AndThen(void (*resume)(int8_t *), LLCLAsyncChainRef chain,
                             int8_t *hdl) {
  unwrap(chain).andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Execute a coroutine and block the current routine until it is complete.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_ExecuteAndWait(void (*resume)(int8_t *), int8_t *hdl,
                                    LLCLRuntimeRef rt,
                                    LLCLAsyncChainRef chain) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(unwrap(chain));
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_ExecuteAndResume(void (*resume)(int8_t *), int8_t *execHdl,
                                      LLCLAsyncChainRef chain,
                                      LLCLRuntimeRef rt, int8_t *resumeHdl) {
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
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_GetCurrentRuntime() {
  return wrap(Runtime::getCurrentRuntimeOrNull());
}

/// Create an LLCL runtime and return its pointer.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(ssize_t numThreads,
                                              const char *profileFilenamePtr,
                                              ssize_t profileFilenameLen) {
  StringRef profileFilename{profileFilenamePtr,
                            static_cast<size_t>(profileFilenameLen)};
  // Create non global runtimes from mojo with mainWillDonate=false. Refer to
  // Runtime.h for detailed explanation.
  auto options = numThreads > 0 ? RuntimeOptions().withMainWillNotDonate()
                                : RuntimeOptions();
  std::unique_ptr<Runtime> runtime = createNestedRuntime(
      options.withNumThreads(numThreads).withProfileFilename(profileFilename));
  return wrap(runtime.release());
}

/// Create an LLCL runtime and return its pointer.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntime(ssize_t numThreads) {
  return KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(numThreads, nullptr, 0);
}

/// Given a pointer to an LLCL runtime, destroy it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_DestroyRuntime(LLCLRuntimeRef rt) {
  delete &unwrap(rt);
}

/// Given a pointer to an LLCL runtime, get the number of threads in it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint32_t
KGEN_CompilerRT_LLCL_ParallelismLevel(LLCLRuntimeRef rt) {
  return unwrap(rt).getWorkQueue()->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// CUDA
//===----------------------------------------------------------------------===//

/// Returns the CUDA stream for the caller's thread, which may have been
/// established by the C++ runtime for the kernel call, or may be null.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT CUstream
KGEN_CompilerRT_LLCL_GetCurrentStream() {
  return CUDA::Globals::getCurrentStreamInTLS();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CudaContextSetDevice(void *devCtx,
                                     MojoValue::DestructorFn destructor,
                                     LLCLWrapper<CUDA::CUDARuntime> runtime) {
  unwrap(runtime).deviceContext = MojoValue(devCtx, destructor);
}

//===----------------------------------------------------------------------===//
// MojoCallContext
//===----------------------------------------------------------------------===//

/// Emplaces the chain in the given call context.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_MojoCallContext_Complete(
    LLCLMojoCallContextRef callContext) {
  unwrap(callContext).complete();
}

/// Sets the chain in the given call context to be an error.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_MojoCallContext_SetToError(
    LLCLMojoCallContextRef callContext, const char *messagePtr,
    ssize_t messageLen) {
  StringRef message(messagePtr, messageLen);
  unwrap(callContext).setToError(message);
}

/// Get the cuda stream from the context. Null for cpu kernels.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_LLCL_MojoCallContext_GetCUStream(
    LLCLMojoCallContextRef callContext) {
  return unwrap(callContext).cuStream;
}

/// Get cuda device from cuda runtime.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_LLCL_MojoCallContext_GetCudaDevice(
    LLCLMojoCallContextRef callContext) {
  auto runtime = unwrap(callContext).deviceRuntime;
  auto &deviceContext =
      reinterpret_cast<CUDA::CUDARuntime *>(runtime)->deviceContext;
  assert(deviceContext.has_value());
  return deviceContext.value().getData();
}

//===----------------------------------------------------------------------===//
// Packing functions for creating async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateAsync_bool(bool data, LLCLWrapper<AnyAsyncValueRef> async,
                                 LLCLWrapper<Runtime> runtimePtr) {
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
                                   LLCLWrapper<AnyAsyncValueRef> async,
                                   LLCLWrapper<Runtime> runtimePtr) {
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
KGEN_CompilerRT_CreateAsyncVoidStar(void *data,
                                    LLCLWrapper<AnyAsyncValueRef> async,
                                    LLCLWrapper<Runtime> runtimePtr) {
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
KGEN_CompilerRT_CreateAsyncBufferRef(void *data, size_t size,
                                     LLCLWrapper<AnyAsyncValueRef> async,
                                     LLCLWrapper<Runtime> runtimePtr) {
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
KGEN_CompilerRT_CreateAsyncTensorSpec(ssize_t *data, ssize_t rank,
                                      int8_t rawDType,
                                      LLCLWrapper<AnyAsyncValueRef> async,
                                      LLCLWrapper<Runtime> runtimePtr) {
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

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_CreateOwnedAsyncMojoValue(void *data,
                                          void (*destructorFn)(void *),
                                          LLCLWrapper<AnyAsyncValueRef> async,
                                          LLCLWrapper<Runtime> runtimePtr) {
  Runtime &runtime = unwrap(runtimePtr);
  AnyAsyncValueRef &value = unwrap(async);

  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<MojoValue>(data, destructorFn);
  } else {
    assert(!value.isReady());
    value =
        AnyAsyncValueRef::createReady<MojoValue>(runtime, data, destructorFn);
  }
}

//===----------------------------------------------------------------------===//
// Unpacking functions for reading async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetValueFromAsync(LLCLWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    return value.getPointer()->getUnderlyingPtr();
  } else {
    return value.getPointerToData();
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetDataFromBuffer(LLCLWrapper<AnyAsyncValueRef> async,
                                  size_t *sizeOut) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &buffer = value.get<TensorBufferRef>();
  *sizeOut = buffer.getSize();
  return buffer.getBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint8_t
KGEN_CompilerRT_GetTensorSpecFromAsync(ssize_t *data, ssize_t rank,
                                       LLCLWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &spec = value.get<TensorSpec>();
  for (int i = 0; i < rank; ++i)
    data[i] = spec[i];
  return spec.getEltType().getValue();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetContextPayloadPtr(size_t index,
                                     LLCLWrapper<StateContext> rawCtx) {
  StateContext &ctx = unwrap(rawCtx);
  return ctx.getStateSlot(index).getUnderlyingPointer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetContextAndSizeFromAsync(
    size_t *size, LLCLWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &ctx = value.get<StateContext>();
  *size = ctx.getNumStateSlots();
  return reinterpret_cast<void *>(&ctx);
}

//===----------------------------------------------------------------------===//
// SpinWaiter function
//===----------------------------------------------------------------------===//

/// Creates a new SpinWaiter instance.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_LLCL_InitializeSpinWaiter() {
  return new SpinWaiter<true>();
}

/// Waits on the SpinWaiter
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_SpinWaiter_Wait(LLCLSpinWaiterRef waiter) {
  unwrap(waiter).wait();
}

/// Destroys the given SpinWaiter.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_DestroySpinWaiter(LLCLSpinWaiterRef waiter) {
  delete (SpinWaiter<true> *)(waiter.ptr);
}

//===----------------------------------------------------------------------===//
// Strings
//===----------------------------------------------------------------------===//

void M::KGEN::registerLLCL(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_LLCL_InitializeChain",
                   (void *)&KGEN_CompilerRT_LLCL_InitializeChain});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroyChain",
                   (void *)&KGEN_CompilerRT_LLCL_DestroyChain});
  funcs.push_back({"KGEN_CompilerRT_LLCL_Complete",
                   (void *)&KGEN_CompilerRT_LLCL_Complete});
  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_Wait", (void *)&KGEN_CompilerRT_LLCL_Wait});
  funcs.push_back({"KGEN_CompilerRT_LLCL_Wait_Timeout",
                   (void *)&KGEN_CompilerRT_LLCL_Wait_Timeout});

  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_Execute", (void *)&KGEN_CompilerRT_LLCL_Execute});
  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_AndThen", (void *)&KGEN_CompilerRT_LLCL_AndThen});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ExecuteAndWait",
                   (void *)&KGEN_CompilerRT_LLCL_ExecuteAndWait});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ExecuteAndResume",
                   (void *)&KGEN_CompilerRT_LLCL_ExecuteAndResume});

  funcs.push_back({"KGEN_CompilerRT_LLCL_GetCurrentRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_GetCurrentRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile",
                   (void *)&KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile});
  funcs.push_back({"KGEN_CompilerRT_LLCL_CreateRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_CreateRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroyRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_DestroyRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ParallelismLevel",
                   (void *)&KGEN_CompilerRT_LLCL_ParallelismLevel});

  funcs.push_back({"KGEN_CompilerRT_LLCL_GetCurrentStream",
                   (void *)&KGEN_CompilerRT_LLCL_GetCurrentStream});
  funcs.push_back({"KGEN_CompilerRT_CudaContextSetDevice",
                   (void *)&KGEN_CompilerRT_CudaContextSetDevice});
  funcs.push_back({"KGEN_CompilerRT_CreateAsync_ssizet",
                   (void *)&KGEN_CompilerRT_CreateAsync_ssizet});
  funcs.push_back({"KGEN_CompilerRT_CreateAsync_bool",
                   (void *)&KGEN_CompilerRT_CreateAsync_bool});
  funcs.push_back({"KGEN_CompilerRT_CreateAsyncBufferRef",
                   (void *)&KGEN_CompilerRT_CreateAsyncBufferRef});
  funcs.push_back({"KGEN_CompilerRT_CreateAsyncTensorSpec",
                   (void *)&KGEN_CompilerRT_CreateAsyncTensorSpec});
  funcs.push_back({"KGEN_CompilerRT_CreateOwnedAsyncMojoValue",
                   (void *)&KGEN_CompilerRT_CreateOwnedAsyncMojoValue});
  funcs.push_back({"KGEN_CompilerRT_GetValueFromAsync",
                   (void *)&KGEN_CompilerRT_GetValueFromAsync});
  funcs.push_back({"KGEN_CompilerRT_GetTensorSpecFromAsync",
                   (void *)&KGEN_CompilerRT_GetTensorSpecFromAsync});
  funcs.push_back({"KGEN_CompilerRT_GetContextPayloadPtr",
                   (void *)&KGEN_CompilerRT_GetContextPayloadPtr});
  funcs.push_back({"KGEN_CompilerRT_GetContextAndSizeFromAsync",
                   (void *)&KGEN_CompilerRT_GetContextAndSizeFromAsync});
  funcs.push_back({"KGEN_CompilerRT_LLCL_MojoCallContext_Complete",
                   (void *)&KGEN_CompilerRT_LLCL_MojoCallContext_Complete});
  funcs.push_back({"KGEN_CompilerRT_LLCL_MojoCallContext_SetToError",
                   (void *)&KGEN_CompilerRT_LLCL_MojoCallContext_SetToError});
  funcs.push_back({"KGEN_CompilerRT_LLCL_MojoCallContext_GetCUStream",
                   (void *)&KGEN_CompilerRT_LLCL_MojoCallContext_GetCUStream});

  funcs.push_back({"KGEN_CompilerRT_LLCL_InitializeSpinWaiter",
                   (void *)&KGEN_CompilerRT_LLCL_InitializeSpinWaiter});
  funcs.push_back({"KGEN_CompilerRT_LLCL_SpinWaiter_Wait",
                   (void *)&KGEN_CompilerRT_LLCL_SpinWaiter_Wait});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroySpinWaiter",
                   (void *)&KGEN_CompilerRT_LLCL_DestroySpinWaiter});
}
