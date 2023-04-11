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
  AsyncValueRef<Chain> chain;
  LLCL::EncodedLocation loc;
  MojoProfilerEntry profilerEntry;
  SmallVector<LLCL::AnyAsyncValueRef> refs;

  OutputChain(AsyncValueRef<Chain> chain, LLCL::EncodedLocation loc)
      : chain(std::move(chain)), loc(std::move(loc)) {}

  // No implicit copying.
  OutputChain(OutputChain &) = delete;
  OutputChain &operator=(const OutputChain &) = delete;

  OutputChain(OutputChain &&) = default;
  OutputChain &operator=(OutputChain &&) = default;

  // Return copy of this output chain. The chain, location and all refs
  // will be copied into the result. However, the profiling entry will be
  // moved into the result.
  //
  // Called from the Mojo side to establish a heap-allocated version of
  // the OutputChain for a call, which will be deleted on completion.
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

  /// Adds tracing entry with name and detail. The trace begins when the
  /// call is made, and ends when markReady() or markError() are called.
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
  /// Called from the MEF side.
  void emplace() &&;

  /// Set the chain to the given error.
  ///
  /// Called from the MEF side.
  void setToError(Error &&error) &&;

  /// Begin executing the Mojo coroutine pointed to by hdl using the resumption
  /// pointer to by resume.
  void executeAsTask(void (*resume)(int8_t *), int8_t *hdl, size_t taskId);

private:
  void complete();
};

} // namespace M::KGEN

#endif // KGEN_OUTPUTCHAIN_H
