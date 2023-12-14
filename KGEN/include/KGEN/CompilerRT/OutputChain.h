//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILERRT_OUTPUTCHAIN_H
#define KGEN_COMPILERRT_OUTPUTCHAIN_H

#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/GenericUniquePtr.h"
#include "LLCL/Support/Profiling.h"
#include "llvm/ADT/ArrayRef.h"

namespace M::KGEN {

/// Profiling entry for all Mojo tracing.
///
/// Note that the Mojo tracing code does its own filtering based on the
/// kMojo level in MODULAR_LLCL_MAX_PROFILING_LEVEL. However, the level must
/// be at least 1 for any such events to be captured since they are all
/// funneled through this entry.
using MojoProfilerEntry = M::ProfilerEntry<Trace::EnableTrace(Trace::kMojo, 1)>;

/// ########################################################
/// ## CAUTION: About to be renamed KGEN::MojoCallContext ##
/// ########################################################
///
/// An OutputChain conveys the context for a call from the C++ runtime into
/// a Mojo entry point, aka kernel.
///
/// The kernel may be fully synchronous, or may launch sub-tasks or other
/// asynchronous work and return.
///
/// It holds:
///  - An AnyAsyncValueRef 'chain' which the Mojo kernel can use to indicate
///    when the kernel has finished or exited with an error.
///     - For synchronous pure-CPU kernels, the chain is the usual
///       AsyncValueRef<Chain>, which must be completed by markReady or
///       markError before return.
///     - For asynchronous pure-CPU kernels, the chain is again an
///       AsyncValueRef<Chain>, and the kernel may move the chain, launch
///       sub-tasks, and return. The moved chain must then be completed by
///       markReady or markError after all sub-tasks have completed.
///     - For CPU kernels which launch non-CPU work (such as a CUDA kernel),
///       the chain may be a device-type specific AsyncValue. The CPU portion
///       of the kernel may use markError to signal a launch error before
///       returning. Otherwise, the chain is completed by the underlying
///       launch machinery. Mojo kernels may access device-specific properties
///       of the chain (such as a CUDA stream representing the kernel's GPU
///       computation).
///  - An EncodedLocation, which can be used by the markError method.
///  - Any number of AnyAsyncValueRefs and GenericUniquePtrs to keep C++
///    runtime objects alive until the markReady or markError methods are
///    called.
///  - Additional target-specific context, such as a CUDA stream on which to
///    launch all async CUDA operations.
///
/// The Mojo OutputChainPtr struct points to heap-allocated instances of this
/// class.
struct OutputChain {
  /// Chain on which consumers are waiting. The actual representation
  /// may depend on the device executing the 'inner' kernel, if any.
  AnyAsyncValueRef chain;
  /// Location to use for any errors.
  LLCL::EncodedLocation loc;
#if MODULAR_PARANOID
  /// All 'uses' of 'resources' needed by this call which should be considered
  /// active while the call is in flight. This can be used to capture which
  /// of the refs below are for BufferRefs which are being read, written or
  /// modified while the call is active, and thus detect use-after-free and
  /// data races over BufferRefs.
  SmallVector<LLCL::ResourceUse> uses;
#endif

  /// AsyncValue references to hold alive until markReady() or markError()
  /// is called.
  SmallVector<LLCL::AnyAsyncValueRef> refs;
  /// Other odd's 'n end's to keep alive also.
  SmallVector<LLCL::GenericUniquePtr> extras;

  /// For kernel calls using cuda.kernel.execute.via_cpu, the CUDA stream
  /// to use for all launched CUDA kernels and other async operations.
  /// The runtime currently ensures all kernel inputs are correctly
  /// synchronized to this stream, and will ensure all users of the kernel
  /// results will be similarly synchronized.
  ///
  /// Eventually stream management may be moved into Mojo and this field
  /// can be removed.
  void *cudaStream = nullptr;

  OutputChain(AnyAsyncValueRef chain, LLCL::EncodedLocation loc)
      : chain(std::move(chain)), loc(std::move(loc)) {}

  /// No copy or move.
  OutputChain(OutputChain &) = delete;
  OutputChain &operator=(const OutputChain &) = delete;
  OutputChain(OutputChain &&) = default;
  OutputChain &operator=(OutputChain &&) = default;

  ~OutputChain();

  /// Return a 'fork' of this output chain:
  ///  - The chain and location are copied, so are valid in both this and the
  ///    result.
  ///  - The refs and extras are moved into the result, on the assumption the
  ///    caller will create sub-tasks and take responsibility for calling
  ///    markReady/markError when they complete.
  ///  - Trace entries are left behind, and will be ended when this object
  ///    is cleaned up.
  ///
  /// Called from Mojo asynchronous kernels to prepare for executing sub-tasks.
  /// The fork result will constructed into a heap allocated object, and
  /// deleted from the Mojo side when all sub-tasks have completed.
  ///
  /// Synchronous Mojo kernels will not call fork, and instead will work
  /// directly with the output chain passed to them.
  OutputChain fork();

  /// Processes work items until the chain is completed.
  void await();

  /// Assert fail if the underlying chain is not ready.
  void assertReady();

  /// Returns the runtime associated with this output chain.
  LLCL::CompactRuntimePtr getRuntime() const { return chain.getRuntime(); }

  /// Move the AnyAsyncValueRefs in argRef(s) into this OutputChain.
  /// These references will keep their referenced AsyncValues alive until
  /// the OutputChain is completed.
  ///
  /// Called from the C++ runtime.
  void transfer(LLCL::AnyAsyncValueRef argRef);
  void transfer(SmallVector<LLCL::AnyAsyncValueRef> argRefs);

  /// Similarly for extras.
  ///
  /// Called from the C++ runtime.
  void transfer(LLCL::GenericUniquePtr extra);

  /// Indicate the Mojo call is complete.
  ///
  /// Called from the Mojo side.  The chain is not consumed so that we can
  /// always safely await and check for errors on the chain irrespective of
  /// whether the Mojo kernel is asynchronous or synchronous.
  ///
  /// For CPU kernels only.
  void markReady();

  /// Indicate the Mojo call failed with the given message.
  ///
  /// Called from the Mojo side. The chain is not consumed so that we can
  /// always safely await and check for errors on the chain irrespective of
  /// whether the Mojo kernel is asynchronous or synchronous.
  ///
  /// For CPU kernels only.
  void markError(StringRef message);

  /// Emplace the chain.
  ///
  /// Called from the C++ runtime when there's nothing to be done by
  /// a Mojo kernel. The chain is not consumed so that we can
  /// always safely await and check for errors on the chain irrespective of
  /// whether the Mojo kernel is asynchronous or synchronous.
  void emplace();

  /// Set the chain to the given error.
  ///
  /// Called from the C++ runtime when an error is detected before entering
  /// the Mojo kernel. The chain is not consumed so that we can
  /// always safely await and check for errors on the chain irrespective of
  /// whether the Mojo kernel is asynchronous or synchronous.
  void setToError(Error &&error);

  /// Indicates the current task is done for the purposes of task overhang
  /// detections. Only needed for tasks which do not otherwise call markReady()
  /// or markError() to signal their completion. Only significant if build
  /// has enabled task overhang detection.
  void taskIsDone();

  /// For kernel calls using cuda.kernel.execute.via_cpu only: Returns the
  /// CUDA CUstream handle being used to synchronize execution of the launched
  /// CUDA kernel. We'll use a void* to avoid including any CUDA headers.
  void *getCUDAStream() const { return cudaStream; }

private:
  /// Cleanup all resource held by the OutputChain in preparation for emplacing
  /// or setting to error the out chain.
  void complete();
};

} // namespace M::KGEN

#endif // KGEN_COMPILERRT_OUTPUTCHAIN_H
