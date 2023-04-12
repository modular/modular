//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_OUTPUTCHAIN_H
#define KGEN_OUTPUTCHAIN_H

#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/Profiling.h"
#include "llvm/ADT/ArrayRef.h"

namespace M::KGEN {

/// Type of profiling entries for Mojo trace events.
///
/// Note that the Mojo tracing code does it's own filtering based on the
/// kMojo level in MODULAR_LLCL_MAX_PROFILING_LEVEL. However, the level must
/// be at least 1 for any such events to be captured.
using MojoProfilerEntry = M::ProfilerEntry<Trace::EnableTrace(Trace::kMojo, 1)>;

/// An OutputChain helps mediate between asynchronous Mojo kernels and the C++
/// MEF runtime. It is responsible for:
///  - Holding an AsyncValueRef<Chain> which will be either emplaced or
///    set to an error when Mojo code invokes the markReady or markError
///    methods.
///  - Holding an EncodedLocation which can be used by the markError method.
///  - Holding at most one MojoProfilerEntry to be recorded when the markReady
///    or markError methods are called.
///  - Holding any number of AnyAsyncValueRefs to keep buffers or other
///    AsyncValues alive until markReady or markError methods are called.
///
/// The Mojo OutputChainPtr struct points to heap-allocated instances of this
/// class.
///
/// NOTE: We could handle recording any profiling entry and releasing ownership
/// of the additional refs by placing andThenSync waiters on the chain.
/// However, the order in which andThenSync waiter are called when the chain
/// becomes ready is difficult to control. Thus it may be that a consuming
/// operation begins execution when the chain becomes ready *before* either
/// the profiling entry has been recorded or the additional refs have been
/// released. By taking responsibility here we guarantee ordering.
struct OutputChain {
  /// Chain on which consumers are waiting.
  AsyncValueRef<Chain> chain;
  /// Location to use for any errors.
  LLCL::EncodedLocation loc;
  /// The profiler entry for the kernel execution. Begins when the
  /// kernel is called, and ends when either the kernel calls markReady()/
  /// markError(), or the kernel returns to the MEF executor.
  ///
  /// For synchronous kernels, this profiler entry will capture the true
  /// work of the kernel. The kernel will not have launched any sub-tasks.
  ///
  /// For asynchronous kernels, this profiler entry will live only while the
  /// kernel establishes its sub-tasks, and will be recorded when the kernel
  /// returns to MEF.
  MojoProfilerEntry profilerEntry;
  /// The 'prototype' profiler entry to be used when the Mojo kernel calls
  /// executeAsTask. Each task will append '.task' to the profile name,
  /// and some task id details to the profile details. This entry, however,
  /// is never recorded.
  MojoProfilerEntry prototypeProfilerEntry;
  /// AsyncValue references to hold alive until markReady() or markError()
  /// is called.
  SmallVector<LLCL::AnyAsyncValueRef> refs;

  OutputChain(AsyncValueRef<Chain> chain, LLCL::EncodedLocation loc)
      : chain(std::move(chain)), loc(std::move(loc)) {}

  // No implicit copying.
  OutputChain(OutputChain &) = delete;
  OutputChain &operator=(const OutputChain &) = delete;

  OutputChain(OutputChain &&) = default;
  OutputChain &operator=(OutputChain &&) = default;

  // Return copy of this output chain. The chain, location,
  // prototypeProfilerEntry and all refs are moved into the result.
  // However, the profilerEntry is not moved since it will be used on the
  // MEF side.
  //
  // Called from the Mojo side to take over ownership of the OutputChain as
  // a heap-allocated object which can live until markReady() or markError()
  // are called. Only used by asynchronous kernels, synchronous kernels will
  // work directly with the output chain passed to them.
  OutputChain copy();

  // Processes work items until the chain is completed.
  // FOR USE BY TESTING AND SEARCH ONLY.
  void await();

  /// Returns the runtime associated with this output chain.
  LLCL::CompactRuntimePtr getRuntime() const { return chain.getRuntime(); }

  /// Move the AnyAsyncValueRefs in argRef(s) into this OutputChain.
  /// These references will keep their referenced AsyncValues alive until
  /// the OutputChain is completed.
  ///
  /// Called from the MEF side.
  void transfer(LLCL::AnyAsyncValueRef &&argRef);
  void transfer(SmallVector<LLCL::AnyAsyncValueRef> &&argRefs);

  /// Adds tracing entry with name and detail.
  ///
  /// Called from the Mojo side.
  void trace(StringRef name, StringRef detail);

  /// Indicate the Mojo call is complete.
  ///
  /// Called from the Mojo side.
  void markReady();

  /// Indicate the Mojo call failed with the given message.
  ///
  /// Called from the Mojo side.
  void markError(StringRef message);

  /// Emplace the chain.
  ///
  /// Called from the MEF side when there's nothing to be done by
  /// a Mojo kernel.
  void emplace() &&;

  /// Set the chain to the given error.
  ///
  /// Called from the MEF side when an error is detected before entering
  /// the Mojo kernel.
  void setToError(Error &&error) &&;

  /// Record any profiling entry if it has not been recorded already.
  ///
  /// Called from the MEF side when control returns. The kernel may have
  /// launched sub-tasks, which will continue to execute asynchronously.
  void recordProfilerEntry() &&;

  /// Begin executing the Mojo coroutine pointed to by hdl using the resumption
  /// pointer to by resume.
  void executeAsTask(void (*resume)(int8_t *), int8_t *hdl, size_t taskId);

private:
  void complete();
};

} // namespace M::KGEN

#endif // KGEN_OUTPUTCHAIN_H
