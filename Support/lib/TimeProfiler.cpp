//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/TimeProfiler.h"

#include "Config/Version.h"
#include "Support/Globals/GlobalProfilerContext.h"
#include "Support/Host.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

using namespace M;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::time_point_cast;

using ClockType = ProfilerEntry<true>::ClockType;
using TimePointType = ProfilerEntry<true>::TimePointType;
using DurationType = duration<ClockType::rep, ClockType::period>;

namespace M {

struct TimeTraceThreadProfiler {
  explicit TimeTraceThreadProfiler(unsigned timeTraceGranularity)
      : tid(llvm::get_threadid()), timeTraceGranularity(timeTraceGranularity) {
    llvm::get_thread_name(threadName);
  }

  /// Start a new entry.
  void begin(ProfilerEntry<true> &&entry) {
    stack.emplace_back(std::move(entry));
  }

  /// End the current running entry.
  void end() {
    assert(!stack.empty() && "must call begin() first");
    record(std::move(stack.back()));
    stack.pop_back();
  }

  /// Record the given entry.
  void record(ProfilerEntry<true> &&entry) {
    if (entry.name.empty())
      return;

    if (entry.end == TimePointType())
      entry.end = ClockType::now();

    // Calculate duration at full precision for overall counts.
    DurationType duration = entry.end - entry.start;

    // Only include sections longer or equal to timeTraceGranularity msec.
    if (duration_cast<microseconds>(duration).count() >= timeTraceGranularity)
      entries.emplace_back(std::move(entry));
  }

  /// The stack of currently running timers.
  SmallVector<ProfilerEntry<true>, 16> stack;

  /// The set of completed timer entries.
  SmallVector<ProfilerEntry<true>, 128> entries;

  /// The name of the thread this profiler is running on.
  SmallString<0> threadName;

  /// The id of the thread this profiler is running on.
  const uint64_t tid;

  // Minimum time granularity (in microseconds)
  const unsigned timeTraceGranularity;
};

/// This class represents the profiler context for a specific thread.
struct ThreadProfilerContext {
  ~ThreadProfilerContext();

  /// Return the profiler instance for this thread, or nullptr if one isn't
  /// active.
  static TimeTraceThreadProfiler *get();

  /// The profiler attached to this thread.
  TimeTraceThreadProfiler *profiler = nullptr;
};

/// This class represents the main context used for profiling.
struct GlobalProfilerContext {
  GlobalProfilerContext(unsigned granularity, StringRef name)
      : timeTraceGranularity(granularity), procName(name),
        pid(llvm::sys::Process::getProcessId()),
        beginningOfTime(system_clock::now()), startTime(ClockType::now()) {}

  /// The minimum time granularity (in microseconds) for time trace profiler.
  unsigned timeTraceGranularity = 0;

  /// The name of the process this profiler is running on.
  StringRef procName;

  /// The id of the process this profiler is running on.
  const llvm::sys::Process::Pid pid;

  /// System clock time when the session was begun.
  time_point<system_clock> beginningOfTime;

  /// Profiling clock time when the session was begun.
  const TimePointType startTime;

  /// Lock used to guard access to the running profilers.
  std::mutex lock;

  /// The set of running profilers for each thread.
  std::vector<std::unique_ptr<TimeTraceThreadProfiler>> profilers;

  /// A set of active thread profiler contexts.
  DenseSet<ThreadProfilerContext *> threadProfilerContexts;

  SmallVector<std::string> inputShapes;
};
} // namespace M

static int reportError(Twine errorMessage) {
  llvm::errs() << errorMessage << "\n";
  return EXIT_FAILURE;
}

TimeTraceThreadProfiler *ThreadProfilerContext::get() {
  static thread_local ThreadProfilerContext instance;
  if (!instance.profiler) {
    if (auto *ctx = Globals::getGlobalProfilerContext()) {
      std::lock_guard<std::mutex> lock(ctx->lock);

      // Add this profiler to the main context.
      instance.profiler =
          ctx->profilers
              .emplace_back(std::make_unique<TimeTraceThreadProfiler>(
                  ctx->timeTraceGranularity))
              .get();
      ctx->threadProfilerContexts.insert(&instance);
    }
  }
  return instance.profiler;
}

ThreadProfilerContext::~ThreadProfilerContext() {
  // The current thread is dying, so try to pass over ownership of the
  // profiler to the main context.
  if (auto *ctx = Globals::getGlobalProfilerContext()) {
    std::lock_guard<std::mutex> lock(ctx->lock);
    ctx->threadProfilerContexts.erase(this);
  }
}

//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

void M::Detail::timeTraceProfilerInitialize(unsigned timeTraceGranularity,
                                            StringRef procName) {
  assert(!Globals::getGlobalProfilerContext() &&
         "profiler should not be initialized");
  Globals::setGlobalProfilerContext(new GlobalProfilerContext(
      timeTraceGranularity, llvm::sys::path::filename(procName)));

  // Prep the profiler for the main thread.
  ThreadProfilerContext::get();
}

void M::Detail::timeTraceProfilerDestroy() {
  assert(Globals::getGlobalProfilerContext() &&
         "profiler should be initialized");
  if (auto *ctx = Globals::exchangeGlobalProfilerContext(
          nullptr)) { // Clear out any dangling pointers in thread profiler
                      // contexts.
    {
      std::lock_guard<std::mutex> guard(ctx->lock);
      for (auto *tpc : ctx->threadProfilerContexts)
        tpc->profiler = nullptr;
    }
    delete ctx;
  }
}

void M::Detail::timeTraceProfilerAddInputShape(const std::string &shape) {
  assert(Globals::getGlobalProfilerContext() &&
         "profiler should be initialized");
  Globals::getGlobalProfilerContext()->inputShapes.push_back(shape);
}

//===----------------------------------------------------------------------===//
// Trace Output
//===----------------------------------------------------------------------===//

// Output JSON format is documented here
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview

void M::Detail::timeTraceProfilerWriteTrace(llvm::raw_pwrite_stream &os) {
  assert(Globals::getGlobalProfilerContext() &&
         "profiler should be initialized");
  auto &ctx = *Globals::getGlobalProfilerContext();
  std::lock_guard<std::mutex> lock(ctx.lock);
  auto profilers = llvm::make_pointee_range(ctx.profilers);
  assert(llvm::all_of(profilers,
                      [](const auto &ttp) { return ttp.stack.empty(); }) &&
         "all profiler sections should be ended when calling write");

  // For visualization purposes only.
  // Sometimes callers can have the same start time (in ns) as their callees.
  // Since events are push to Entries when the trace event ends, callees
  // appear before callers in Entries. When Perfetto sees 2 events with the
  // same start time, it displays the first one (callee) above the second one
  // (caller) which is not what we want. After reversing Entries, callers
  // appear before callees, and therefore callers appear above callees in the
  // profiler.
  for (TimeTraceThreadProfiler &ttp : profilers)
    std::reverse(ttp.entries.begin(), ttp.entries.end());

  llvm::json::OStream jsonOS(os);
  jsonOS.objectBegin();
  jsonOS.attributeBegin("traceEvents");
  jsonOS.arrayBegin();

  // Emit all events for the main flame graph.
  auto writeEvent = [&](const ProfilerEntry<true> &event, uint64_t tid) {
    auto startUs = event.getFlameGraphStartUs(ctx.startTime);
    auto durUs = event.getFlameGraphDurUs();
    jsonOS.object([&] {
      jsonOS.attribute("pid", ctx.pid);
      jsonOS.attribute("tid", int64_t(tid));
      jsonOS.attribute("ph", event.isSampling() ? "C" : "X");
      jsonOS.attribute("ts", startUs);
      jsonOS.attribute("dur", durUs);
      jsonOS.attribute("name", event.name);
      if (event.isSampling()) {
        jsonOS.attributeObject(
            "args", [&]() { jsonOS.attribute("value", event.getValue()); });
      } else {
        const std::string &detail = event.getDetail();
        if (!detail.empty()) {
          jsonOS.attributeObject("args",
                                 [&]() { jsonOS.attribute("detail", detail); });
        }
      }
    });
  };
  for (const TimeTraceThreadProfiler &ttp : profilers)
    for (const ProfilerEntry<true> &entry : ttp.entries)
      writeEvent(entry, ttp.tid);

  auto writeMetadataEvent = [&](const char *name, uint64_t tid, StringRef arg) {
    jsonOS.object([&] {
      jsonOS.attribute("cat", "");
      jsonOS.attribute("pid", ctx.pid);
      jsonOS.attribute("tid", int64_t(tid));
      jsonOS.attribute("ts", 0);
      jsonOS.attribute("ph", "M");
      jsonOS.attribute("name", name);
      jsonOS.attributeObject("args", [&] { jsonOS.attribute("name", arg); });
    });
  };

  writeMetadataEvent("process_name", ctx.pid, ctx.procName);
  for (const TimeTraceThreadProfiler &ttp : profilers)
    writeMetadataEvent("thread_name", ttp.tid, ttp.threadName);

  jsonOS.arrayEnd();
  jsonOS.attributeEnd();

  // Emit the absolute time when time profiling started. This can be used to
  // combine the profiling data from multiple processes and preserve actual time
  // intervals.
  jsonOS.attribute("beginningOfTime",
                   time_point_cast<microseconds>(ctx.beginningOfTime)
                       .time_since_epoch()
                       .count());

  // Emit input tensor info
  jsonOS.attributeBegin("tensorInfo");
  jsonOS.objectBegin();
  jsonOS.attributeBegin("inputShapes");
  jsonOS.arrayBegin();
  for (auto &shape : ctx.inputShapes)
    jsonOS.value(shape);
  jsonOS.arrayEnd();
  jsonOS.attributeEnd();
  jsonOS.objectEnd();
  jsonOS.attributeEnd();

  // Emit software version info
  jsonOS.attributeBegin("versionInfo");
  jsonOS.objectBegin();
  ModularVersion version = getModularVersion();
  jsonOS.attribute("modular-git-sha", version.revision);
  jsonOS.attribute("modular-build-type", version.buildType);
  std::ostringstream profilingLevelOctal;
  profilingLevelOctal << std::oct << "0" << MODULAR_LLCL_MAX_PROFILING_LEVEL;
  jsonOS.attribute("modular-profiling-level", profilingLevelOctal.str());
  jsonOS.objectEnd();
  jsonOS.attributeEnd();

  // Emit the host machine info, if we can retrieve it.
  auto hostMachineInfoOr = getHostMachineInfo();
  if (hostMachineInfoOr.isError()) {
    reportError("warning: time-profiler failed to "
                "retrieve system-info for tracefile");
  } else {
    jsonOS.attributeBegin("hostMachineInfo");
    hostMachineInfoOr.takeValue().print(jsonOS);
    jsonOS.attributeEnd();
  }

  jsonOS.objectEnd();
  os.flush();
}

//===----------------------------------------------------------------------===//
// Event Stream Output
//===----------------------------------------------------------------------===//

namespace {

/// A more convenient representation for the event stream output.
struct Event {
  DurationType start;
  uint64_t tid;
  DurationType dur;
  std::string name;
  std::string detail;
  DurationType end;
  bool isBegin;

  Event() = default;

  /// Event representing time trace profiling entry
  Event(const ProfilerEntry<true> &entry, uint64_t tid, TimePointType startTime)
      : start(entry.start - startTime), tid(tid), dur(entry.end - entry.start),
        name(entry.name), end(entry.end - startTime), isBegin(true) {
    if (entry.isSampling())
      detail = std::to_string(entry.getValue());
    else
      detail = entry.getDetail();
  }

  /// Event representing the 'end' of this event.
  Event toEnd() {
    Event result = *this;
    result.start = end;
    result.isBegin = false;
    return result;
  }

  bool operator<(const Event &rhs) const {
    return std::tie(start, tid, name, detail) <
           std::tie(rhs.start, rhs.tid, rhs.name, rhs.detail);
  }

  void write(llvm::raw_pwrite_stream &os) const {
    os << llvm::format("%6d  %10d  ", tid,
                       duration_cast<microseconds>(start).count());
    os << (isBegin ? "BEG  " : "END  ");
    os << llvm::format("%10d  ", duration_cast<microseconds>(dur).count());
    os << name;
    if (!detail.empty())
      os << "/" << detail;
    os << "\n";
  }
};
} // namespace

void M::Detail::timeTraceProfilerWriteEventStream(llvm::raw_pwrite_stream &os) {
  assert(Globals::getGlobalProfilerContext() &&
         "profiler should be initialized");
  auto &ctx = *Globals::getGlobalProfilerContext();
  std::lock_guard<std::mutex> lock(ctx.lock);

  std::vector<Event> events;
  for (const TimeTraceThreadProfiler &ttp :
       llvm::make_pointee_range(ctx.profilers)) {
    for (const ProfilerEntry<true> &e : ttp.entries) {
      events.emplace_back(e, ttp.tid, ctx.startTime);
      events.emplace_back(events.back().toEnd());
    }
  }
  std::sort(events.begin(), events.end());

  os << "Thread   Start(us)  B/E     Dur(us)  Name/Detail\n";
  os << "------  ----------  ---  ----------  ------------------------------\n";
  for (const auto &event : events)
    event.write(os);
  os.flush();
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

ErrorOrSuccess M::Detail::timeTraceProfilerWrite(StringRef preferredFileName,
                                                 StringRef fallbackFileName) {
  assert(Globals::getGlobalProfilerContext() &&
         "profiler should be initialized");

  // Set up filename base.
  std::string path = preferredFileName.str();
  if (path.empty())
    path = fallbackFileName == "-" ? "out" : fallbackFileName.str();

  std::error_code ec;

  {
    // Write time trace.
    std::string tracePath = path == "-" ? path : path + ".time-trace";
    llvm::raw_fd_ostream os(tracePath, ec, llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + tracePath + "(" +
                   Twine(ec.message()) + ")");
    timeTraceProfilerWriteTrace(os);
  }

  {
    // Write the raw event stream.
    std::string eventStreamPath =
        path == "-" ? path : path + ".time-events.txt";
    llvm::raw_fd_ostream os(eventStreamPath, ec,
                            llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + eventStreamPath + "(" +
                   Twine(ec.message()) + ")");
    timeTraceProfilerWriteEventStream(os);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TimeTraceProfilerEntry
//===----------------------------------------------------------------------===//

std::string ProfilerEntry<true>::toImmediateDebugString() {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << llvm::get_threadid() << "  ";
  os << (end == TimePointType() ? "BEG  " : "END  ");
  os << name;
  if (isTiming()) {
    const std::string &detail = getDetail();
    if (!detail.empty()) {
      os << "/" << detail;
    }
  } else {
    os << "/" << getValue();
  }
  os << "\n";
  return str;
}

//===----------------------------------------------------------------------===//
// Public interface to the ThreadProfilerContext's TimeTraceThreadProfiler.
//===----------------------------------------------------------------------===//

void M::Detail::timeTraceProfilerBegin(ProfilerEntry<true> &&entry) {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->begin(std::move(entry));
}

void M::Detail::timeTraceProfilerEnd() {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->end();
}

bool M::Detail::timeTraceProfilerIsActive() {
  return ThreadProfilerContext::get();
}

void M::Detail::timeTraceProfilerRecord(ProfilerEntry<true> &&entry) {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->record(std::move(entry));
}
