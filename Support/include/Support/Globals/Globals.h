//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_GLOBALS_GLOBALS_H
#define SUPPORT_GLOBALS_GLOBALS_H

#include "Support/SymbolExport.h"

#include "llvm/ADT/StringMap.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace M {
namespace Detail {
class TypeInfoTable;
}

namespace ProfilingDetail {
struct GlobalProfilerContext;
}

namespace Profiling::Detail {
struct RangeSink;
}

namespace Globals {

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::ProfilingDetail::GlobalProfilerContext *ctx);

extern MODULAR_CXX_EXPORT Detail::TypeInfoTable &
getTypeInfoTableSingleton(const std::function<Detail::TypeInfoTable *()> &ctor);

// Process-wide storage for `Config::setGlobalValue()` overrides. These
// live in `libMSupportGlobals.so` so that writes from one shared library
// (e.g. the `max._core` Python extension that wraps `DebugConfig`) are
// visible to reads from another (e.g. `libmax.so` which contains
// `GraphCompiler/FrameworkFrontend`). A function-local static in
// `Support/lib/Configuration.cpp` would give each consumer a distinct
// copy, breaking cross-library propagation.
//
// Callers must hold `getConfigOverridesMutex()` while reading or writing
// `getConfigOverrides()`.
extern MODULAR_CXX_EXPORT std::mutex &getConfigOverridesMutex();
extern MODULAR_CXX_EXPORT llvm::StringMap<std::string> &getConfigOverrides();

// Process-wide hot-path state for //Support:ProfilingRanges. The range shim
// is statically linked into many shared libraries, so this state lives here
// for the same reason as the config overrides above: a gate flip or sink
// attachment performed through one shim copy must be visible to every other
// copy's hot path, and per-copy function-local statics would break that.
// Only the shim (Ranges.cpp) and the profiler integration write these —
// the integration receives the gate addresses through the plugin ABI's
// registerShim (Support/Profiling/PluginABI.h); everything else reads them
// through the inline accessors in Support/Profiling/Ranges.h.
struct ProfilingRangeGlobals {
  // Explicit enable()/disable() intent.
  std::atomic<bool> enabled{false};
  // True while a trace of either origin is live and ranges record.
  std::atomic<bool> recordingActive{false};
  // The attached profiler integration's function table, set when a shim
  // copy's acquire first succeeds (racing copies may each store their own
  // equivalent table; last write wins); valid for the process lifetime.
  std::atomic<const M::Profiling::Detail::RangeSink *> cachedSink{nullptr};
};

extern MODULAR_CXX_EXPORT ProfilingRangeGlobals &getProfilingRangeGlobals();

// Process-wide state for //Support:ProfilingRanges' capture labels
// (Support/Profiling/CaptureLabels.h). Op spans push labels from the Mojo
// runtime's copy of the shim while the device launch site that reads them
// lives in another shared library, so both must reach one instance.
struct CaptureLabelOp {
  uint64_t opSeq = 0;
  uint64_t parentSeq = 0;
  uint64_t thread = 0;
  std::string label;
};

struct CaptureLabelGlobals {
  // Number of device-graph captures in progress; labels record while > 0.
  std::atomic<int> activeCaptures{0};
  // Source of per-push op sequence numbers and per-thread ids; 0 means none.
  std::atomic<uint64_t> nextOpSeq{1};
  std::atomic<uint64_t> nextThread{1};
  // Every label pushed while a capture is in progress, in push order.
  // Guarded by `opsMutex`; cleared when the last capture ends.
  std::mutex opsMutex;
  std::vector<CaptureLabelOp> ops;
};

struct CaptureLabelFrame {
  std::string label;
  uint64_t opSeq = 0;
  // The thread that pushed the op; differs from the holder for an adopted op.
  uint64_t thread = 0;
};

struct CaptureLabelThreadState {
  uint64_t thread = 0;
  std::vector<CaptureLabelFrame> stack;
};

extern MODULAR_CXX_EXPORT CaptureLabelGlobals &getCaptureLabelGlobals();
extern MODULAR_CXX_EXPORT CaptureLabelThreadState &getCaptureLabelThreadState();

// Per-thread request correlation, described in
// `Support/include/Support/RequestContext.h`.
extern MODULAR_CXX_EXPORT std::optional<int64_t> &getRequestBatchId();

} // namespace Globals

} // namespace M

#endif // SUPPORT_GLOBALS_GLOBALS_H
