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

/// AsyncChainRef is an opaque wrapper around an `AsyncValueRef<Chain>`.
struct AsyncChainRef {
  void *storage;
};

static inline AsyncValueRef<Chain> &unwrap(AsyncChainRef ptr) {
  return *reinterpret_cast<AsyncValueRef<Chain> *>(ptr.storage);
}

/// LLCLRuntimeRef is an opaque wrapper around an `LLCL::Runtime`.
using LLCLRuntimeRef = void *;

static inline Runtime *unwrap(LLCLRuntimeRef ptr) {
  return reinterpret_cast<Runtime *>(ptr);
}
static inline LLCLRuntimeRef wrap(const Runtime *ptr) {
  return reinterpret_cast<LLCLRuntimeRef>(const_cast<Runtime *>(ptr));
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Given the async context of a coroutine, initialize its token value. The
/// runtime pointer must have already been set.
extern "C" void KGEN_CompilerRT_LLCL_InitializeChain(LLCLRuntimeRef rt,
                                                     AsyncChainRef chain) {
  new (&unwrap(chain))
      AsyncValueRef<Chain>(AsyncValueRef<Chain>::allocate(unwrap(rt)));
}

/// Given the async context of a coroutine, destroy its token value.
extern "C" void KGEN_CompilerRT_LLCL_DestroyChain(AsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine.
extern "C" void KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *),
                                             int8_t *hdl,
                                             LLCLRuntimeRef runtime) {
  unwrap(runtime)->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
}

/// Resume a coroutine when the current one completes.
extern "C" void KGEN_CompilerRT_LLCL_AndThen(void (*resume)(int8_t *),
                                             AsyncChainRef chain, int8_t *hdl) {
  unwrap(chain).getPointer()->andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Block until the coroutine is done.
extern "C" void KGEN_CompilerRT_LLCL_Wait(AsyncChainRef chain) {
  await(unwrap(chain));
}

/// Execute a coroutine and block the current routine until it is complete.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndWait(void (*resume)(int8_t *),
                                                    int8_t *hdl,
                                                    LLCLRuntimeRef rt,
                                                    AsyncChainRef chain) {
  unwrap(rt)->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(unwrap(chain));
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
extern "C" void KGEN_CompilerRT_LLCL_ExecuteAndResume(void (*resume)(int8_t *),
                                                      int8_t *execHdl,
                                                      AsyncChainRef chain,
                                                      LLCLRuntimeRef rt,
                                                      int8_t *resumeHdl) {
  unwrap(rt)->getWorkQueue()->addTask([resume, execHdl]() { resume(execHdl); });
  unwrap(chain).getPointer()->andThenAsync(
      [resumeHdl, resume]() { resume(resumeHdl); });
}

/// Given the async context of a coroutine, indicate that it is complete by
/// setting its token value.
extern "C" void KGEN_CompilerRT_LLCL_Complete(AsyncChainRef chain) {
  unwrap(chain).copy().emplace();
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
  delete unwrap(ptr);
}

/// Given a compact pointer to an LLCL runtime, get the number of threads in it.
extern "C" uint32_t KGEN_CompilerRT_LLCL_ParallelismLevel(LLCLRuntimeRef ptr) {
  return unwrap(ptr)->getWorkQueue()->getParallelismLevel();
}
