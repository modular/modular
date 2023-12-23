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
///  - An AnyAsyncValueRef 'chain' which is currently only used to support
///    the markError method.
///  - An EncodedLocation, which can be used by the markError method.
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

  /// No copy, but move is ok.
  OutputChain(OutputChain &) = delete;
  OutputChain &operator=(const OutputChain &) = delete;
  OutputChain(OutputChain &&) = default;
  OutputChain &operator=(OutputChain &&) = default;

  /// Returns the runtime associated with this output chain.
  LLCL::CompactRuntimePtr getRuntime() const { return chain.getRuntime(); }

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
  void markError(StringRef message);

  /// Set the chain to the given error.
  ///
  /// Called from the C++ runtime when an error is detected before entering
  /// the Mojo kernel. The chain is not consumed so that we can
  /// always safely await and check for errors on the chain irrespective of
  /// whether the Mojo kernel is asynchronous or synchronous.
  void setToError(Error &&error);

  /// For kernel calls using cuda.kernel.execute.via_cpu only: Returns the
  /// CUDA CUstream handle being used to synchronize execution of the launched
  /// CUDA kernel. We'll use a void* to avoid including any CUDA headers.
  void *getCUDAStream() const { return cudaStream; }
};

} // namespace M::KGEN

#endif // KGEN_COMPILERRT_OUTPUTCHAIN_H
