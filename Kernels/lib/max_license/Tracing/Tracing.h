//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KERNELS_TRACING_H
#define KERNELS_TRACING_H

#include "Support/ErrorOr.h"
#include "Support/max_license/Tracy/TracyZone.h"
#include "tracing_c_api.h"
#include "llvm/ADT/StringRef.h"

namespace M {

using MojoTracePtr = std::unique_ptr<void, decltype(&destroy_trace)>;

/// Class for starting and stopping profiling via tracing.mojo.
class MojoTrace {
public:
  /// Fallible constructor for the trace object.
  static ErrorOr<MojoTrace> create(StringRef message, StringRef color = "");

  /// Defaults to null Mojo trace, which is safe to call destroy_trace on.
  MojoTrace() : mojoTrace(MojoTracePtr(nullptr, &destroy_trace)) {}

  /// Initializes this interface to a Mojo trace object.
  explicit MojoTrace(MojoTracePtr trace) : mojoTrace(std::move(trace)) {}

  /// Return opaque pointer to mojo type.
  void *opaque() const { return mojoTrace.get(); }

  /// Begins the profiling span.
  void start() { start_trace(mojoTrace.get()); }

  /// Completes the profiling span.
  void end() { end_trace(mojoTrace.get()); }

  /// Marks the profile at the current time. This is not a time span and not
  /// all profilers support this.
  void mark() { mark_trace(mojoTrace.get()); }

  /// Checks whether the Mojo trace object was successfully created.
  operator bool() const { return mojoTrace.get(); }

private:
  /// A void pointer to the Mojo `MojoTrace` struct.
  MojoTracePtr mojoTrace;
};

/// Class for RAII-style tracing.
/// Starts and ends a trace on construction/destruction.
/// Does nothing if the trace event had an error.
///
/// Example usage:
/// ```c++
/// void exampleFunction() {
///   // Begin scoped trace with message "ComputeTask" and optional color
///   // "blue".
///   ScopedMojoTrace trace(/*message=*/"ComputeTask", /*color=*/"blue");
///   // ... perform work that will be traced ...
/// } // trace.end() called automatically when going out of scope
/// ```
class ScopedMojoTrace {
public:
  ScopedMojoTrace(StringRef message, StringRef color = "") {
#ifdef TRACY_ENABLE
    const char *funcPtr =
        message.empty() ? "M::ScopedMojoTrace" : message.data();
    size_t funcLen = message.empty() ? sizeof("M::ScopedMojoTrace") - 1
                                     : static_cast<size_t>(message.size());
    tracyCtx = tracyZoneBegin("ScopedMojoTrace", sizeof("ScopedMojoTrace") - 1,
                              funcPtr, funcLen, message.data(),
                              static_cast<size_t>(message.size()),
                              parseTracyColor(color));
#else
    ErrorOr<MojoTrace> traceOr = MojoTrace::create(message, color);
    if (!traceOr.isError()) {
      trace = std::move(*traceOr);
      trace.start();
    }
#endif // TRACY_ENABLE
  }

  ~ScopedMojoTrace() noexcept {
#ifdef TRACY_ENABLE
    tracyZoneEnd(tracyCtx);
#else
    if (trace)
      trace.end();
#endif // TRACY_ENABLE
  }

private:
  MojoTrace trace;
#ifdef TRACY_ENABLE
  uint64_t tracyCtx = 0;
#endif // TRACY_ENABLE
};

} // namespace M

#endif // KERNELS_TRACING_H
