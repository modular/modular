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
  /// This is a compact runtime pointer. When executing an async coroutine from
  /// a synchronous context, a runtime pointer must be explicitly provided. The
  /// runtime pointer is implicitly propagated through nested async coroutines.
  LLCL::CompactRuntimePtr runtime;
};

/// C binding unwrapper for `AsyncContext`.
inline AsyncContext *unwrap(int8_t *ptr) {
  return reinterpret_cast<AsyncContext *>(ptr);
}

// Enforce invariants between the LLCL C shim and KGEN.
static_assert(sizeof(LLCL::CompactRuntimePtr) == 1,
              "expected CompactRuntimePtr to be 1 byte");
static_assert(sizeof(AsyncValueRef<Chain>) == sizeof(void *),
              "expected AsyncValueRef to be the size of a pointer");
} // namespace

/// Given the async context of a coroutine, initialize its token value. The
/// runtime pointer must have already been set.
extern "C" void KGEN_CompilerRT_LLCL_InitializeContext(int8_t *asyncCtx) {
  AsyncContext *ctx = unwrap(asyncCtx);
  new (&ctx->chain)
      AsyncValueRef<Chain>(AsyncValueRef<Chain>::allocate(ctx->runtime));
}

/// Given the async context of a coroutine, destroy its token value.
extern "C" void KGEN_CompilerRT_LLCL_DestroyContext(int8_t *asyncCtx) {
  unwrap(asyncCtx)->chain.~AsyncValueRef<Chain>();
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine.
extern "C" void KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *),
                                             int8_t *hdl, int8_t runtime) {
  LLCL::CompactRuntimePtr::getFromOpaqueToken(runtime)->getWorkQueue()->addTask(
      [resume, hdl] { resume(hdl); });
}

/// Resume a coroutine when the currenet one completes.
extern "C" void KGEN_CompilerRT_LLCL_AndThen(void (*resume)(int8_t *),
                                             int8_t *asyncCtx, int8_t *hdl) {
  AsyncContext *ctx = unwrap(asyncCtx);
  ctx->chain.getPointer()->andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Block until the coroutine is done.
extern "C" void KGEN_CompilerRT_LLCL_Wait(int8_t *asyncCtx) {
  AsyncContext *ctx = unwrap(asyncCtx);
  LLCL::await(ctx->chain);
}

/// Execute a coroutine and block the current routine until it is complete.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndWait(void (*resume)(int8_t *),
                                                    int8_t *hdl,
                                                    int8_t *asyncCtx) {
  AsyncContext *ctx = unwrap(asyncCtx);
  ctx->runtime->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  LLCL::await(ctx->chain);
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndResume(void (*resume)(int8_t *),
                                                      int8_t *execHdl,
                                                      int8_t *asyncCtx,
                                                      int8_t *resumeHdl) {
  AsyncContext *ctx = unwrap(asyncCtx);
  ctx->runtime->getWorkQueue()->addTask(
      [resume, execHdl]() { resume(execHdl); });
  ctx->chain.getPointer()->andThenAsync(
      [resumeHdl, resume]() { resume(resumeHdl); });
}

/// Given the async context of a coroutine, indicate that it is complete by
/// setting its token value.
extern "C" void KGEN_CompilerRT_LLCL_Complete(int8_t *asyncCtx) {
  unwrap(asyncCtx)->chain.copy().emplace();
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// Create an LLCL runtime and return it as a compact pointer.
extern "C" int8_t
KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(ssize_t numThreads,
                                              const char *profileFilenamePtr,
                                              ssize_t profileFilenameLen) {
  StringRef profileFilename{profileFilenamePtr,
                            static_cast<size_t>(profileFilenameLen)};
  auto *runtime = new LLCL::Runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createThreadPoolWorkQueue(numThreads, {}, !profileFilename.empty()),
      profileFilename);
  AsyncValue::registerType<LLCL::Chain>();
  return runtime->getCompactPtr().getAsOpaqueToken();
}

/// Create an LLCL runtime and return it as a compact pointer.
extern "C" int8_t KGEN_CompilerRT_LLCL_CreateRuntime(ssize_t numThreads) {
  return KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(numThreads, nullptr, 0);
}

/// Given a compact pointer to an LLCL runtime, destroy it.
extern "C" void KGEN_CompilerRT_LLCL_DestroyRuntime(int8_t ptr) {
  delete LLCL::CompactRuntimePtr::getFromOpaqueToken(ptr).get();
}

/// Given a compact pointer to an LLCL runtime, get the number of threads in it.
extern "C" uint32_t KGEN_CompilerRT_LLCL_ParallelismLevel(int8_t ptr) {
  return LLCL::CompactRuntimePtr::getFromOpaqueToken(ptr)
      ->getWorkQueue()
      ->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// TaskGroup
//===----------------------------------------------------------------------===//

extern "C" void KGEN_CompilerRT_LLCL_AddTaskToGroup(
    int8_t *tg, ssize_t (*tgCounterDecr)(void *), int8_t *taskCtx) {
  unwrap(taskCtx)->chain.andThenAsync([tg, tgCounterDecr] {
    if (tgCounterDecr(tg) != 0)
      return;
    unwrap(tg)->chain.copy().emplace();
  });
}
