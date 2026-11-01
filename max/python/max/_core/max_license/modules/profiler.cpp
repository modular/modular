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

#include "Support/Configuration.h"
#include "Support/Profiling/Ranges.h"
#include "_core/Graph/Bindings.h"
#include "_core/Profiler/Trace.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace M::Profiler::Python;

namespace {

// IDs of the ranges range_begin_with_id opened on this thread, innermost
// last. The profiler's range stack is shared with other callers on the
// thread, so range_end pops only when its ID is on top of this one.
// Function-local because a namespace-scope thread_local vector needs a
// global destructor (-Wglobal-constructors).
std::vector<uint64_t> &openRangeIds() {
  static thread_local std::vector<uint64_t> ids;
  return ids;
}

std::atomic<uint64_t> gNextRangeId{1};

// Serializes start and stop, so that a stop's temporary output path and its
// lastTraceError() read can't interleave with another start or stop.
std::mutex &captureMutex() {
  static std::mutex mutex;
  return mutex;
}

// True from a startCapture that took the profiler until the stopCapture that
// ends it. Guarded by captureMutex().
bool gOwnsCapture = false;

// True while a capture started by enable() is live, as opposed to a trace an
// external daemon drives.
bool isCaptureRecording() {
  return M::Profiling::isEnabled() && M::Profiling::isRangeRecordingActive();
}

// Overrides a Config key for one scope and then restores the caller's
// override, or unsets the key if there was none.
class ScopedConfigOverride {
public:
  ScopedConfigOverride(llvm::StringRef key, llvm::StringRef value) : key(key) {
    // MODULAR_DEBUG sets overrides too; parse it first so they're restored.
    M::Config::parseModularDebugEnv();
    previous = M::Config::getGlobalValueIfSet(key);
    M::Config::setGlobalValue(key, value);
  }
  ~ScopedConfigOverride() {
    if (previous)
      M::Config::setGlobalValue(key, *previous);
    else
      M::Config::unsetGlobalValue(key);
  }

private:
  std::string key;
  std::optional<std::string> previous;
};

bool startCapture() {
  std::lock_guard<std::mutex> lock(captureMutex());
  if (!M::Profiling::haveProfiler())
    return false;
  // Takes only an idle profiler, so a capture never ends or redirects one it
  // didn't start: enable() would cancel a Dynolog on-demand trace, and
  // stopCapture would end the session's capture. Those start without
  // captureMutex(), so one starting during this call can still lose.
  if (M::Profiling::isEnabled() || M::Profiling::isRangeRecordingActive())
    return false;
  // enable() latches the session's step window; a capture started here runs
  // until stopCapture instead.
  ScopedConfigOverride warmup("max-debug.profiling-warmup-steps", "0");
  ScopedConfigOverride active("max-debug.profiling-active-steps", "0");
  M::Profiling::enable();
  gOwnsCapture = true;
  return true;
}

std::string stopCapture(const std::string &outputPath) {
  std::lock_guard<std::mutex> lock(captureMutex());
  if (!std::exchange(gOwnsCapture, false))
    return "start_capture() started no capture";
  const bool wasRecording = isCaptureRecording();
  {
    // disable() writes to the configured output path; the override ends
    // with the save so it doesn't leak into later session-level captures.
    ScopedConfigOverride path("max-debug.profiling-output-path", outputPath);
    // Also clears a start that is still waiting for a device context.
    M::Profiling::disable();
    M::Profiling::waitForTrace();
  }
  if (!wasRecording)
    return "no capture was recording";
  return M::Profiling::lastTraceError();
}

} // namespace

void M::Graph::Python::bindProfilerModule(nb::module_ m) {
  m.doc() = R"nbdoc(
      Provides Python bindings for the MAX profiler.
  )nbdoc";

  nb::class_<PyMojoTrace>(m, "Trace", R"nbdoc(
      A context manager for creating profiling spans.

      Examples:
          >>> with Trace("foo", color="modular_purple"):
          >>>   # Run `bar()` inside the profiling span.
          >>>   bar()
          >>> # The profiling span ends when the context manager exits.

      Args:
          message: The name of the span.
          color: The color of the span.
  )nbdoc")
      .def(nb::init<const char *, const char *>(), nb::arg("message"),
           nb::arg("color") = "modular_purple")
      .def("__enter__", &PyMojoTrace::enter,
           // Reference the parent Trace object without taking ownership.
           nb::rv_policy::reference_internal, R"nbdoc(
          Begins a profiling event.
      )nbdoc")
      .def("__exit__", &PyMojoTrace::exit, nb::arg("exc_type") = nb::none(),
           nb::arg("exc_value") = nb::none(), nb::arg("traceback") = nb::none(),
           R"nbdoc(
          Ends a profiling event.
      )nbdoc")
      .def("mark", &PyMojoTrace::mark,
           R"nbdoc(
          Marks an event in the trace timeline.
      )nbdoc");

  m.def("is_profiling_enabled", &isProfilingEnabled, R"nbdoc(
      Returns whether profiling is enabled.
  )nbdoc");

  m.def("set_gpu_profiling_state", &setGPUProfilingState, R"nbdoc(
      Sets the GPU profiling state.
  )nbdoc");

  m.def(
      "range_begin_with_id",
      [](uint64_t correlation_id, std::string_view name) -> uint64_t {
        if (!M::Profiling::isRangeRecordingActive())
          return 0;
        M::Profiling::rangeBeginWithId(correlation_id, name, /*color=*/0);
        uint64_t id = gNextRangeId.fetch_add(1, std::memory_order_relaxed);
        openRangeIds().push_back(id);
        return id;
      },
      nb::arg("correlation_id"), nb::arg("name"),
      R"nbdoc(
      Begins a CPU range whose libkineto ``External id`` is ``correlation_id``.

      Kernels launched on this thread inside the range, outside any nested
      range, carry the ID. Work launched from other threads, such as AsyncRT
      workers, relates to it only by time. The ID is stored in 32 bits, and
      libkineto omits an ID of 0.

      Pass the result to :func:`range_end` on the same thread.

      Args:
          correlation_id: Caller-supplied join key, such as a batch ID.
          name: The range's label in the trace.

      Returns:
          A nonzero range ID, or 0 if no libkineto trace is live, in which
          case nothing is recorded.
  )nbdoc");

  m.def(
      "range_end",
      [](uint64_t range_id) {
        if (range_id == 0)
          return;
        std::vector<uint64_t> &open = openRangeIds();
        if (open.empty() || open.back() != range_id) {
          if (PyErr_WarnEx(PyExc_RuntimeWarning,
                           "range_end() ignored: the range is not the "
                           "innermost range_begin_with_id range open on "
                           "this thread",
                           1) != 0)
            throw nb::python_error();
          return;
        }
        open.pop_back();
        M::Profiling::rangeEnd();
      },
      nb::arg("range_id"), R"nbdoc(
      Ends a range that :func:`range_begin_with_id` opened.

      Does nothing for an ID of 0. Otherwise warns and ends nothing unless
      the range is the innermost one :func:`range_begin_with_id` opened on
      this thread. Ranges opened inside it by other APIs must end first.

      Args:
          range_id: The ID :func:`range_begin_with_id` returned.
  )nbdoc");

  m.def("load_profiler_plugin", &M::Profiling::haveProfiler, R"nbdoc(
      Loads the profiler plugin if it can be found.

      On AMD, call it before any device is created; loaded later, the plugin
      can't record.

      Returns:
          True if a profiler is available.
  )nbdoc");

  m.def("start_capture", &startCapture,
        nb::call_guard<nb::gil_scoped_release>(), R"nbdoc(
      Starts a libkineto capture if the profiler is idle.

      Call it on a thread with a current device context, such as the thread
      that created the devices; otherwise the capture doesn't start until a
      device is next initialized. The capture runs until :func:`stop_capture`,
      ignoring the session's profiling step-window settings. Blocks while a
      :func:`stop_capture` is saving.

      Returns:
          True if this call started a capture. False if no profiler is
          available, or if a capture or a Dynolog on-demand trace is already
          running, which this call leaves alone.
  )nbdoc");

  m.def("stop_capture", &stopCapture, nb::arg("output_path"),
        nb::call_guard<nb::gil_scoped_release>(), R"nbdoc(
      Stops the capture :func:`start_capture` started, saving its trace.

      Leaves any other capture running, such as one the session's profiling
      config started. May be called from any thread.

      Args:
          output_path: File to write the trace to, expanded like
              ``MODULAR_MAX_DEBUG_PROFILING_OUTPUT_PATH``.

      Returns:
          An empty string on success, otherwise why the trace is missing or
          incomplete.
  )nbdoc");

  m.def(
      "request_burst",
      [](std::string_view reason) {
        // The scheduler calls this for every prefill batch; keep the
        // profiling-off path to the one check, without a GIL round trip.
        if (!M::Profiling::isEnabled())
          return false;
        nb::gil_scoped_release release;
        return M::Profiling::requestBurst(reason);
      },
      nb::arg("reason"), R"nbdoc(
      Requests an extra step-window burst for an event, opened now.

      Applies only to a capture with ``profiling-period-steps`` set, where it
      adds a burst between scheduled ones for an event the schedule would
      likely miss, such as a prefill batch. Call it just before the execute
      the burst should start with. The burst covers ``profiling-active-steps``
      steps and is saved to ``<output>.event<k>.json``, labeled with
      ``reason``. At most one request is honored per gap between scheduled
      bursts, and none before the first scheduled burst.

      Args:
          reason: Why the burst was requested, recorded in the trace.
              Characters outside ``[A-Za-z0-9_.-]`` become ``_``.

      Returns:
          ``True`` if a burst opened. ``False`` when profiling is off, which
          costs a single check, or when the request was not honored.
  )nbdoc");

  m.def("is_capture_recording", &isCaptureRecording, R"nbdoc(
      Returns whether a capture is recording.

      Covers a capture from :func:`start_capture` or the session's profiling
      config.
  )nbdoc");
}
