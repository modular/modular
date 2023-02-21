//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

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

struct SpinWaiterRef {
  void *storage;
};

static inline SpinWaiter<> &unwrap(SpinWaiterRef ptr) {
  return *reinterpret_cast<SpinWaiter<> *>(ptr.storage);
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
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_InitializeChain(LLCLRuntimeRef rt,
                                                         AsyncChainRef chain) {
  assert(checkTypeIdsAreCoherent(*unwrap(rt)) && "type ids are not coherent");
  new (&unwrap(chain))
      AsyncValueRef<Chain>(takeRCRef(AsyncValue::allocate<Chain>(unwrap(rt))));
}

/// Given the async context of a coroutine, destroy its token value.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_DestroyChain(AsyncChainRef chain) {
  unwrap(chain).~AsyncValueRef<Chain>();
}

MODULAR_EXPORT void KGEN_CompilerRT_LLCL_InitWaiter(SpinWaiterRef waiter) {
  new (&unwrap(waiter)) SpinWaiter<>();
}

MODULAR_EXPORT void KGEN_CompilerRT_LLCL_WaiterWait(SpinWaiterRef waiter) {
  unwrap(waiter).wait();
}

//===----------------------------------------------------------------------===//
// Coroutine / Future
//===----------------------------------------------------------------------===//

/// Execute a coroutine.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_Execute(void (*resume)(int8_t *),
                                                 int8_t *hdl,
                                                 LLCLRuntimeRef runtime) {
  unwrap(runtime)->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
}

/// Resume a coroutine when the current one completes.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_AndThen(void (*resume)(int8_t *),
                                                 AsyncChainRef chain,
                                                 int8_t *hdl) {
  unwrap(chain).andThenAsync([hdl, resume]() { resume(hdl); });
}

/// Block until the coroutine is done.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_Wait(AsyncChainRef chain) {
  await(unwrap(chain));
}

/// Execute a coroutine and block the current routine until it is complete.
MODULAR_EXPORT void
KGEN_CompilerRT_LLCL_ExecuteAndWait(void (*resume)(int8_t *), int8_t *hdl,
                                    LLCLRuntimeRef rt, AsyncChainRef chain) {
  unwrap(rt)->getWorkQueue()->addTask([resume, hdl] { resume(hdl); });
  await(unwrap(chain));
}

/// Execute a coroutine. Register a completion handler to resume another
/// coroutine when the scheduled coroutine completes.
MODULAR_EXPORT void
KGEN_CompilerRT_LLCL_ExecuteAndResume(void (*resume)(int8_t *), int8_t *execHdl,
                                      AsyncChainRef chain, LLCLRuntimeRef rt,
                                      int8_t *resumeHdl) {
  unwrap(rt)->getWorkQueue()->addTask([resume, execHdl]() { resume(execHdl); });
  unwrap(chain).andThenAsync([resumeHdl, resume]() { resume(resumeHdl); });
}

/// Given the async context of a coroutine, indicate that it is complete by
/// setting its token value.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_Complete(AsyncChainRef chain) {
  unwrap(chain).copy().emplace();
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// Create an LLCL runtime and return it as a compact pointer.
MODULAR_EXPORT LLCLRuntimeRef KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(
    ssize_t numThreads, const char *profileFilenamePtr,
    ssize_t profileFilenameLen) {
  StringRef profileFilename{profileFilenamePtr,
                            static_cast<size_t>(profileFilenameLen)};
  auto *runtime =
      new Runtime(createLeakCheckAllocator(createMallocAllocator()),
                  createThreadPoolWorkQueue(numThreads), profileFilename);
  return wrap(runtime);
}

/// Create an LLCL runtime and return it as a compact pointer.
MODULAR_EXPORT LLCLRuntimeRef
KGEN_CompilerRT_LLCL_CreateRuntime(ssize_t numThreads) {
  return KGEN_CompilerRT_LLCL_CreateRuntimeWithProfile(numThreads, nullptr, 0);
}

/// Given a compact pointer to an LLCL runtime, destroy it.
MODULAR_EXPORT void KGEN_CompilerRT_LLCL_DestroyRuntime(LLCLRuntimeRef ptr) {
  delete unwrap(ptr);
}

/// Given a compact pointer to an LLCL runtime, get the number of threads in it.
MODULAR_EXPORT uint32_t
KGEN_CompilerRT_LLCL_ParallelismLevel(LLCLRuntimeRef ptr) {
  return unwrap(ptr)->getWorkQueue()->getParallelismLevel();
}

void M::KGEN::registerLLCL(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_LLCL_InitializeChain",
                   (void *)&KGEN_CompilerRT_LLCL_InitializeChain});
  funcs.push_back({"KGEN_CompilerRT_LLCL_DestroyChain",
                   (void *)&KGEN_CompilerRT_LLCL_DestroyChain});
  funcs.push_back({"KGEN_CompilerRT_LLCL_InitWaiter",
                   (void *)&KGEN_CompilerRT_LLCL_InitWaiter});
  funcs.push_back({"KGEN_CompilerRT_LLCL_WaiterWait",
                   (void *)&KGEN_CompilerRT_LLCL_WaiterWait});
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
}
