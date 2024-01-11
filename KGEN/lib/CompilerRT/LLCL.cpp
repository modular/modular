//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CUDASupport/Globals/Globals.h"
#include "KGEN/CompilerRT/OutputChain.h"
#include "KGEN/CompilerRT/Registration.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
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
using LLCLOutputChainRef = LLCLWrapper<OutputChain>;
using LLCLAsyncChainRef = LLCLWrapper<AsyncValueRef<Chain>>;

/// Dummy entry point to force loading.
/// (All the other entry points use LLCLWrapper which we don't want to
/// have to include in the header).
COMPILERRT_EXPORT void KGEN_CompilerRT_LLCL_Dummy() {}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Given the async context of a coroutine, initialize its token value. The
/// runtime pointer must have already been set.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_InitializeChain(LLCLRuntimeRef rt,
                                     LLCLAsyncChainRef chain) {
  checkUniqueRuntime(unwrap(rt));
  new (&unwrap(chain))
      AsyncValueRef<Chain>(takeRCRef(AsyncValue::allocate<Chain>(unwrap(rt))));
}

/// Given the async context of a coroutine, destroy its token value.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_DestroyChain(LLCLAsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
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

/// Block until the coroutine is done.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Wait(LLCLAsyncChainRef chain) {
  await(unwrap(chain));
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

/// Given the async context of a coroutine, indicate that it is complete by
/// setting its token value.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Complete(LLCLAsyncChainRef chain) {
#if MODULAR_PARANOID
  unwrap(chain).getRuntime()->getWorkQueue()->taskIsDone();
#endif
  unwrap(chain).copy().emplace();
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
  std::unique_ptr<Runtime> runtime =
      createNestedRuntime(RuntimeOptions()
                              .withNumThreads(numThreads)
                              .withProfileFilename(profileFilename));
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

//===----------------------------------------------------------------------===//
// OutputChainPtr
//===----------------------------------------------------------------------===//

/// Sets an error message on outChain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError(LLCLOutputChainRef outChain,
                                              const char *messagePtr,
                                              ssize_t messageLen) {
  std::string message(messagePtr, messageLen);
  unwrap(outChain).markError(message);
}

void M::KGEN::registerLLCL(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_LLCL_InitializeChain",
                   (void *)&KGEN_CompilerRT_LLCL_InitializeChain});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroyChain",
                   (void *)&KGEN_CompilerRT_LLCL_DestroyChain});
  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_Execute", (void *)&KGEN_CompilerRT_LLCL_Execute});
  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_AndThen", (void *)&KGEN_CompilerRT_LLCL_AndThen});
  funcs.push_back(
      {"KGEN_CompilerRT_LLCL_Wait", (void *)&KGEN_CompilerRT_LLCL_Wait});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ExecuteAndWait",
                   (void *)&KGEN_CompilerRT_LLCL_ExecuteAndWait});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ExecuteAndResume",
                   (void *)&KGEN_CompilerRT_LLCL_ExecuteAndResume});
  funcs.push_back({"KGEN_CompilerRT_LLCL_Complete",
                   (void *)&KGEN_CompilerRT_LLCL_Complete});
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
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError});
}
