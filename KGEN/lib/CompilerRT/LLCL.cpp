//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

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

/// Execute a coroutine.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *), int8_t *hdl,
                             LLCLRuntimeRef rt) {
  checkUniqueRuntime(unwrap(rt));
  unwrap(rt).getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
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
  unwrap(chain).copy().emplace();
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// Create an LLCL runtime and return it as a compact pointer.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(ssize_t numThreads,
                                              const char *profileFilenamePtr,
                                              ssize_t profileFilenameLen) {
  StringRef profileFilename{profileFilenamePtr,
                            static_cast<size_t>(profileFilenameLen)};
  auto *runtime =
      new Runtime(createLeakCheckAllocator(createMallocAllocator()),
                  createThreadPoolWorkQueue(numThreads), profileFilename);
  return wrap(runtime);
}

/// Create an LLCL runtime and return it as a compact pointer.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntime(ssize_t numThreads) {
  return KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(numThreads, nullptr, 0);
}

/// Given a compact pointer to an LLCL runtime, destroy it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_DestroyRuntime(LLCLRuntimeRef rt) {
  delete &unwrap(rt);
}

/// Given a compact pointer to an LLCL runtime, get the number of threads in it.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint32_t
KGEN_CompilerRT_LLCL_ParallelismLevel(LLCLRuntimeRef rt) {
  return unwrap(rt).getWorkQueue()->getParallelismLevel();
}

//===----------------------------------------------------------------------===//
// OutputChainPtr and OwningOutputChainPtr
//===----------------------------------------------------------------------===//

/// Returns outChains's runtime.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_OutputChainPtr_GetRuntime(LLCLOutputChainRef outChain) {
  return wrap(unwrap(outChain).getRuntime().get());
}

/// Emplaces outChain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_MarkReady(LLCLOutputChainRef outChain) {
  unwrap(outChain).markReady();
}

/// Sets an error message on outChain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError(LLCLOutputChainRef outChain,
                                              const char *messagePtr,
                                              ssize_t messageLen) {
  std::string message(messagePtr, messageLen);
  unwrap(outChain).markError(message);
}

/// Returns an empty OutputChain, with empty chain and 'unknown' location.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLOutputChainRef
KGEN_CompilerRT_LLCL_OutputChainPtr_CreateEmpty(LLCLRuntimeRef rt) {
  auto chain = AsyncValueRef<Chain>::allocate(unwrap(rt));
  EncodedLocation loc = LLCL::UnknownLocationDecoder::getEncodedLocation();
  return wrap(new OutputChain(std::move(chain), std::move(loc)));
}

/// Returns a fresh OutputChain who's contents is copied from outChain.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT LLCLOutputChainRef
KGEN_CompilerRT_LLCL_OutputChainPtr_CreateFork(LLCLOutputChainRef outChain) {
  return wrap(new OutputChain(unwrap(outChain).fork()));
}

/// Destroys outChain, which must be the result of a CreateEmpty or
/// CreateMoved.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_Destroy(LLCLOutputChainRef outChain) {
  delete &unwrap(outChain);
}

/// Processes work items until outChain is ready.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_Await(LLCLOutputChainRef outChain) {
  unwrap(outChain).await();
}

/// Assert fail if outChain is not ready.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_AssertReady(LLCLOutputChainRef outChain) {
  unwrap(outChain).assertReady();
}

/// Begins a profiling entry with name and detail when called, and ends it
/// when outChain is completed. If an entry already exists, merge the name
/// and details.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_Trace(LLCLOutputChainRef outChain,
                                          const char *namePtr, ssize_t nameLen,
                                          const char *detailPtr,
                                          ssize_t detailLen) {
  StringRef name(namePtr, nameLen);
  if (detailPtr) {
    StringRef detail(detailPtr, detailLen);
    unwrap(outChain).trace(name, detail);
  } else {
    unwrap(outChain).trace(name);
  }
}

/// Execute a coroutine.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_ExecuteAsTask(LLCLOutputChainRef outChain,
                                                  void (*resume)(int8_t *),
                                                  int8_t *hdl, size_t taskId,
                                                  bool useGlobalQueue) {
  unwrap(outChain).getRuntime()->getWorkQueue()->addTask(
      [taskId, resume, hdl]() mutable {
        // Create a tracing entry for the sub-task, using the last created
        // trace entry as its 'parent'. That could either be the last
        // traced item from a Mojo tracing statement, or the overall profiling
        // entry created by the GraphRT primitive which launched the kernel.
        TimeTraceScope scope(MojoProfilerEntry::createWithCurrentAsParent(
            StringLiteral("task"), (uint64_t)taskId));
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
      useGlobalQueue ? -1 : (int)taskId);
}

/// Indicates the caller's task is done for the purposes of task overhang
/// detection.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_LLCL_OutputChainPtr_TaskIsDone(LLCLOutputChainRef outChain) {
#if MODULAR_PARANOID
  unwrap(outChain).taskIsDone();
#endif
}

/// Returns the CUstream handle held by the OutputChain's underlying
/// AsyncValueRef<CUDAStream>. Only valid for GPU kernels which are
/// launched via a CPU kernel shim. The CUstream is returned as a void*
/// to avoid depending on any CUDA headers.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_LLCL_OutputChainPtr_GetCUDAStream(LLCLOutputChainRef outChain) {
  return unwrap(outChain).getCUDAStream();
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
  funcs.push_back({"KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile",
                   (void *)&KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile});
  funcs.push_back({"KGEN_CompilerRT_LLCL_CreateRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_CreateRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroyRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_DestroyRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_ParallelismLevel",
                   (void *)&KGEN_CompilerRT_LLCL_ParallelismLevel});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_GetRuntime",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_GetRuntime});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_MarkReady",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_MarkReady});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_MarkError});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_CreateEmpty",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_CreateEmpty});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_CreateFork",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_CreateFork});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_Destroy",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_Destroy});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_Await",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_Await});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_AssertReady",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_AssertReady});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_Trace",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_Trace});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_ExecuteAsTask",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_ExecuteAsTask});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_TaskIsDone",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_TaskIsDone});
  funcs.push_back({"KGEN_CompilerRT_LLCL_OutputChainPtr_GetCUDAStream",
                   (void *)&KGEN_CompilerRT_LLCL_OutputChainPtr_GetCUDAStream});
}
