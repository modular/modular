//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"

using namespace M;
using namespace M::LLCL;

//===----------------------------------------------------------------------===//
// AsyncContext
//===----------------------------------------------------------------------===//

namespace {
/// The async context contains information for marshalling the execution of
/// coroutines on an LLCL runtime. This context is stored in the coroutine
/// promise.
struct alignas(8) AsyncContext {
  /// This is the async token that indicates whether the coroutine is done. When
  /// the coroutine is done, its results are available in its promise memory.
  /// The lifetime of the reference is tied to the lifetime of the coroutine.
  AsyncValueRef<Chain> chain;
  /// This is a runtime pointer. When executing an async coroutine from
  /// a synchronous context, a runtime pointer must be explicitly provided. The
  /// runtime pointer is implicitly propagated through nested async coroutines.
  Runtime *runtime;
};

// Enforce invariants between the LLCL C shim and KGEN.
static_assert(sizeof(AsyncValueRef<Chain>) == sizeof(void *),
              "expected AsyncValueRef to be the size of a pointer");

} // namespace

/// AsyncContextRef is an opaque wrapper around AsyncContext.
using AsyncContextRef = void *;

/// LLCLRuntimeRef is an opaque wrapper around Runtime.
using LLCLRuntimeRef = void *;

/// C binding unwrapper for `AsyncContextRef` and `LLCLRuntimeRef`.
template <typename T>
static inline T *unwrap(void *ptr) {
  return reinterpret_cast<T *>(ptr);
}

static inline LLCLRuntimeRef wrap(const Runtime *ptr) {
  return reinterpret_cast<LLCLRuntimeRef>(const_cast<Runtime *>(ptr));
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Given the async context of a coroutine, initialize its token value. The
/// runtime pointer must have already been set.
extern "C" void
KGEN_CompilerRT_LLCL_InitializeContext(AsyncContextRef asyncCtx) {
  auto *ctx = unwrap<AsyncContext>(asyncCtx);
  new (&ctx->chain)
      AsyncValueRef<Chain>(AsyncValueRef<Chain>::allocate(ctx->runtime));
}

/// Given the async context of a coroutine, destroy its token value.
extern "C" void KGEN_CompilerRT_LLCL_DestroyContext(AsyncContextRef asyncCtx) {
  unwrap<AsyncContext>(asyncCtx)->chain.~AsyncValueRef<Chain>();
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine.
extern "C" void KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *),
                                             int8_t *hdl,
                                             LLCLRuntimeRef runtime) {
  unwrap<Runtime>(runtime)->getWorkQueue()->addTask(
      [resume, hdl] { resume(hdl); });
}

/// Resume a coroutine when the current one completes.
extern "C" void KGEN_CompilerRT_LLCL_AndThen(void (*resume)(int8_t *),
                                             AsyncContextRef asyncCtx,
                                             int8_t *hdl) {
  auto *ctx = unwrap<AsyncContext>(asyncCtx);
  ctx->chain.getPointer()->andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Block until the coroutine is done.
extern "C" void KGEN_CompilerRT_LLCL_Wait(AsyncContextRef asyncCtx) {
  auto *ctx = unwrap<AsyncContext>(asyncCtx);
  await(ctx->chain);
}

/// Execute a coroutine and block the current routine until it is complete.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndWait(void (*resume)(int8_t *),
                                                    int8_t *hdl,
                                                    AsyncContextRef asyncCtx) {
  auto *ctx = unwrap<AsyncContext>(asyncCtx);
  ctx->runtime->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(ctx->chain);
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndResume(void (*resume)(int8_t *),
                                                      int8_t *execHdl,
                                                      AsyncContextRef asyncCtx,
                                                      int8_t *resumeHdl) {
  auto *ctx = unwrap<AsyncContext>(asyncCtx);
  ctx->runtime->getWorkQueue()->addTask(
      [resume, execHdl]() { resume(execHdl); });
  ctx->chain.getPointer()->andThenAsync(
      [resumeHdl, resume]() { resume(resumeHdl); });
}

/// Given the async context of a coroutine, indicate that it is complete by
/// setting its token value.
extern "C" void KGEN_CompilerRT_LLCL_Complete(AsyncContextRef asyncCtx) {
  unwrap<AsyncContext>(asyncCtx)->chain.copy().emplace();
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// Create an LLCL runtime and return it as a compact pointer.
extern "C" LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(ssize_t numThreads,
                                              const char *profileFilenamePtr,
                                              ssize_t profileFilenameLen) {
  StringRef profileFilename{profileFilenamePtr,
                            static_cast<size_t>(profileFilenameLen)};
  auto *runtime = new Runtime(
      createLeakCheckAllocator(createMallocAllocator()),
      createThreadPoolWorkQueue(numThreads, {}, !profileFilename.empty()),
      profileFilename);
  AsyncValue::registerType<Chain>();
  return wrap(runtime);
}

/// Create an LLCL runtime and return it as a compact pointer.
extern "C" LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntime(ssize_t numThreads) {
  return KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(numThreads, nullptr, 0);
}

/// Given a compact pointer to an LLCL runtime, destroy it.
extern "C" void KGEN_CompilerRT_LLCL_DestroyRuntime(LLCLRuntimeRef ptr) {
  delete unwrap<Runtime>(ptr);
}

/// Given a compact pointer to an LLCL runtime, get the number of threads in it.
extern "C" uint32_t KGEN_CompilerRT_LLCL_ParallelismLevel(LLCLRuntimeRef ptr) {
  return unwrap<Runtime>(ptr)->getWorkQueue()->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// TaskGroup
//===----------------------------------------------------------------------===//

extern "C" void
KGEN_CompilerRT_LLCL_AddTaskToGroup(AsyncContextRef tg,
                                    ssize_t (*tgCounterDecr)(AsyncContextRef),
                                    AsyncContextRef taskCtx) {
  unwrap<AsyncContext>(taskCtx)->chain.andThenAsync(
      [tg, tgCounterDecr,
       resultChain = unwrap<AsyncContext>(tg)->chain.copy()]() mutable {
        if (tgCounterDecr(tg))
          return;
        std::move(resultChain).emplace();
      });
}
